#!/usr/bin/env python
import os
# CRITICAL FOR DEBUGGING CUDA ERRORS:
# Make CUDA errors synchronous and provide more accurate stack traces.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Attempt to enable detailed device-side assertions for more informative CUDA error messages.
# Note: Setting this in the shell (export TORCH_USE_CUDA_DSA=1) before running is often more reliable.
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import logging
import sys
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json
import numpy as np # For NaN/Inf checks in validation
from tqdm import tqdm
import pandas as pd

# --- Imports for RFDETR and Evaluation ---
try:
    from rfdetr import RFDETRLarge
except ImportError as e:
    user_home = os.path.expanduser("~")
    conda_env_path_segment = os.path.join(user_home, "conda", "envs")
    if conda_env_path_segment in sys.path or any(conda_env_path_segment in p for p in sys.path):
        print(f"INFO: Attempting to import RFDETRLarge from a Conda environment context.")
    print(f"ERROR: Could not import RFDETRLarge. Ensure 'rfdetr' library is installed in your active Python environment ({sys.executable}) and is in your PYTHONPATH if installed elsewhere. Details: {e}")
    sys.exit(1)

import supervision as sv
from supervision.metrics import MeanAveragePrecision


# --- General Configuration ---
NUM_FOLDS = 5
BASE_DATA_PATH_TEMPLATE = "/blue/hulcr/gmarais/PhD/IBBI_work/phase_1_data/2_object_detection_phase_2/coco_rfdetr/coco/cv_iteration_{}"
BASE_RESULTS_DIR = "./results_rfdetr_large_eval"
MODEL_DESCRIPTOR = "RFDETRLarge"
TARGET_GPU_ID_FOR_SEQUENTIAL_RUN = 0
DEBUG_FORCE_CPU = False # <<< SET TO TRUE TO FORCE CPU FOR DEBUGGING INITIALIZATION >>>

RFDETR_TRAIN_PARAMS = {
    "epochs": 3,
    "batch_size": 1,
    "grad_accum_steps": 32,
    "lr": 1e-4,
    "checkpoint_interval": 1,
}

# --- Main Script Logging Setup ---
main_orchestration_log_dir = os.path.join(BASE_RESULTS_DIR, "orchestration_logs")
os.makedirs(main_orchestration_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(main_orchestration_log_dir, f"cv_orchestration_main_sequential_{MODEL_DESCRIPTOR.lower()}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

### --- MODIFICATION START --- ###
# 1. ADDED HELPER FUNCTION TO GET CLASS COUNT FROM COCO FILE
def get_num_classes_from_coco(json_file_path):
    """Reads a COCO annotation file and returns the number of classes."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        if 'categories' in data and isinstance(data['categories'], list):
            # This is the most reliable way: count the defined categories.
            return len(data['categories'])
        else:
            # If no 'categories' key, infer from max annotation ID.
            # This is a fallback and assumes zero-indexed, contiguous IDs.
            max_id = -1
            if 'annotations' in data and data['annotations']:
                for ann in data['annotations']:
                    if 'category_id' in ann and ann['category_id'] > max_id:
                        max_id = ann['category_id']
            # If max_id is 1, num_classes is 2 (0, 1). If max_id is -1, num_classes is 0.
            return max_id + 1
    except (FileNotFoundError, json.JSONDecodeError):
        return 0 # Return 0 if file is invalid or not found
### --- MODIFICATION END --- ###

def validate_coco_annotations(json_file_path, fold_py_logger):
    fold_py_logger.info(f"Validating COCO annotations file: {json_file_path}")
    if not os.path.exists(json_file_path):
        fold_py_logger.error(f"Annotation file not found for validation: {json_file_path}")
        return True

    has_critical_issues = False
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if 'images' not in data or 'annotations' not in data or 'categories' not in data:
            fold_py_logger.error(f"COCO file {json_file_path} is missing 'images', 'annotations', or 'categories' keys.")
            return True

        images_map = {img['id']: img for img in data['images']}
        all_cat_ids_from_defs = set()
        all_cat_ids_from_anns = set()

        if not isinstance(data.get('categories'), list):
            fold_py_logger.error(f"'categories' key in {json_file_path} is present but not a list.")
            return True # Critical issue if categories key exists but isn't a list
        
        if not data['categories']:
            fold_py_logger.warning(f"No categories defined in 'categories' list in {json_file_path}. "
                                     "Model might rely on these for num_classes. Max category ID check will rely on annotations only.")
        else:
            for cat_idx, cat in enumerate(data['categories']):
                if 'id' not in cat:
                    fold_py_logger.warning(f"Category at index {cat_idx} (name: {cat.get('name', 'N/A')}) in {json_file_path} is missing 'id'.")
                    # This could be an issue if IDs are not contiguous or if model expects them from here.
                else:
                    all_cat_ids_from_defs.add(cat['id'])

        fold_py_logger.info(f"Found {len(data.get('annotations', []))} annotations and {len(data.get('images', []))} images in {json_file_path}.")

        for ann_idx, ann in enumerate(data.get('annotations', [])):
            ann_id_display = ann.get('id', f"internal_idx_{ann_idx}")
            bbox = ann.get('bbox')
            img_id = ann.get('image_id')
            cat_id = ann.get('category_id')

            if cat_id is not None:
                all_cat_ids_from_anns.add(cat_id)
            else:
                fold_py_logger.error(f"  Annotation ID {ann_id_display} (Image ID {img_id}) is missing 'category_id'.")
                has_critical_issues = True

            if bbox is None:
                fold_py_logger.error(f"  Annotation ID {ann_id_display} (Image ID {img_id}) is missing 'bbox'.")
                has_critical_issues = True
                continue

            if not isinstance(bbox, list) or len(bbox) != 4:
                fold_py_logger.error(f"  Annotation ID {ann_id_display} (Image ID {img_id}) has malformed bbox: {bbox}. Expected list of 4 numbers.")
                has_critical_issues = True
                continue

            if img_id is None or img_id not in images_map:
                fold_py_logger.error(f"  Annotation ID {ann_id_display} references missing or invalid image_id: {img_id}.")
                has_critical_issues = True

            x, y, w, h = bbox
            current_ann_problems = []
            if not (isinstance(w, (int, float)) and w > 0):
                current_ann_problems.append(f"width not positive_number (value: {w}, type: {type(w)})")
            if not (isinstance(h, (int, float)) and h > 0):
                current_ann_problems.append(f"height not positive_number (value: {h}, type: {type(h)})")

            if any(not isinstance(coord, (int, float)) or np.isnan(coord) or np.isinf(coord) for coord in bbox):
                current_ann_problems.append(f"NaN, Inf or non-numeric in bbox coordinates: {bbox}")

            if current_ann_problems:
                fold_py_logger.error(f"  Critical issues in Annotation ID {ann_id_display} (Image ID {img_id}): {', '.join(current_ann_problems)}. Bbox: {bbox}")
                has_critical_issues = True
        
        final_cat_ids = all_cat_ids_from_defs.union(all_cat_ids_from_anns)
        if not final_cat_ids and data.get('annotations'):
            fold_py_logger.warning(f"Annotations are present but NO category IDs were found in 'categories' list or in any 'annotation.category_id'. "
                                     "This is highly problematic if the model requires class labels.")
            # Not setting has_critical_issues = True here as some models might theoretically handle "no class" scenario.
            # But it's a big warning.
        elif final_cat_ids:
            max_cat_id = max(final_cat_ids)
            min_cat_id = min(final_cat_ids)
            fold_py_logger.info(f"CATEGORY ID CHECK: IDs found in {json_file_path} range from {min_cat_id} to {max_cat_id} (inclusive). "
                                f"Total unique category IDs found: {len(final_cat_ids)}. ")
            fold_py_logger.warning(f"USER ACTION: Ensure this maximum ID ({max_cat_id}) is compatible with the {MODEL_DESCRIPTOR} model's "
                                     "expected number of classes. If the model expects N classes (typically 0 to N-1), "
                                     f"a max_cat_id of {max_cat_id} implies at least {max_cat_id + 1} classes are needed. "
                                     "A mismatch is a common cause of CUDA errors during model initialization.")
            if min_cat_id < 0:
                fold_py_logger.error(f"Found negative category_id ({min_cat_id}). This is invalid for COCO and will likely cause errors.")
                has_critical_issues = True # Negative category IDs are definitively bad.

        if has_critical_issues:
            fold_py_logger.error(f"Critical issues found in {json_file_path}. Training for this fold may be unstable or fail.")
        else:
            fold_py_logger.info(f"Annotation file {json_file_path} passed basic structural validation checks.")

    except json.JSONDecodeError as e:
        fold_py_logger.error(f"Error decoding JSON from {json_file_path}: {e}")
        return True
    except Exception as e:
        fold_py_logger.error(f"Unexpected error during validation of {json_file_path}: {e}", exc_info=True)
        return True

    return has_critical_issues

def parse_rfdetr_native_metrics(rfdetr_metrics_csv_path, comparable_csv_output_path, fold_py_logger):
    fold_py_logger.info(f"Attempting to parse RFDETR native metrics from: {rfdetr_metrics_csv_path}")
    if not os.path.exists(rfdetr_metrics_csv_path):
        fold_py_logger.warning(f"RFDETR native metrics.csv not found at {rfdetr_metrics_csv_path}. Skipping.")
        return
    try:
        df_rfdetr = pd.read_csv(rfdetr_metrics_csv_path)
        df_comparable = pd.DataFrame()

        if 'epoch' in df_rfdetr.columns:
            df_comparable['epoch'] = df_rfdetr['epoch']
        else:
            df_comparable['epoch'] = np.nan # Or range(len(df_rfdetr)) if appropriate
            fold_py_logger.warning("Column 'epoch' not found in RFDETR metrics.")

        # Fill common metrics, using .get() with a default for robustness
        df_comparable['time'] = df_rfdetr.get('time', np.nan) # Assuming RFDETR might output time per epoch
        df_comparable['train/box_loss'] = df_rfdetr.get('train_loss_bbox', 0) + df_rfdetr.get('train_loss_giou', 0)
        df_comparable['train/cls_loss'] = df_rfdetr.get('train_loss_ce', np.nan)
        df_comparable['train/dfl_loss'] = df_rfdetr.get('train_loss_dfl', np.nan) # If RFDETR outputs this
        df_comparable['metrics/precision(B)'] = df_rfdetr.get('val_precision', np.nan) # Or however RFDETR names it
        df_comparable['metrics/recall(B)'] = df_rfdetr.get('val_recall', np.nan)
        df_comparable['metrics/mAP50(B)'] = df_rfdetr.get('val_map_50', np.nan)
        df_comparable['metrics/mAP50-95(B)'] = df_rfdetr.get('val_map', np.nan) # Often primary mAP
        df_comparable['val/box_loss'] = df_rfdetr.get('val_loss_bbox', 0) + df_rfdetr.get('val_loss_giou', 0)
        df_comparable['val/cls_loss'] = df_rfdetr.get('val_loss_ce', np.nan)
        df_comparable['val/dfl_loss'] = df_rfdetr.get('val_loss_dfl', np.nan) # If RFDETR outputs this
        
        # Learning rates - RFDETR might log one LR or per param group
        common_lr = df_rfdetr.get('lr', np.nan) 
        df_comparable['lr/pg0'] = df_rfdetr.get('lr_pg0', common_lr) # Be specific if RFDETR logs per group
        df_comparable['lr/pg1'] = df_rfdetr.get('lr_pg1', common_lr)
        df_comparable['lr/pg2'] = df_rfdetr.get('lr_pg2', common_lr)

        ultralytics_column_order = [
            'epoch', 'time', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2'
        ]
        # Ensure all columns exist, filling with NaN if not produced by RFDETR
        for col in ultralytics_column_order:
            if col not in df_comparable.columns:
                df_comparable[col] = np.nan
        
        df_comparable = df_comparable[ultralytics_column_order] # Enforce order
        df_comparable.to_csv(comparable_csv_output_path, index=False, float_format='%.5f')
        fold_py_logger.info(f"Saved comparable epoch metrics to: {comparable_csv_output_path}")

    except FileNotFoundError: # Should be caught by os.path.exists, but just in case
        fold_py_logger.warning(f"RFDETR native metrics.csv not found at {rfdetr_metrics_csv_path} during read. Skipping.")
    except pd.errors.EmptyDataError:
        fold_py_logger.warning(f"RFDETR native metrics.csv at {rfdetr_metrics_csv_path} is empty. Skipping.")
    except Exception as e_parse:
        fold_py_logger.error(f"Error parsing RFDETR native metrics from {rfdetr_metrics_csv_path}: {e_parse}", exc_info=True)


def generate_evaluation_outputs(eval_model, val_images_dir, val_annotations_path_for_custom_eval, classes, output_dir, fold_py_logger):
    fold_py_logger.info(f"Starting FINAL model custom evaluation in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        fold_py_logger.info(f"Custom Eval - Using validation images from: {val_images_dir}")
        fold_py_logger.info(f"Custom Eval - Using validation annotations from: {val_annotations_path_for_custom_eval}")

        if not os.path.isdir(val_images_dir):
            fold_py_logger.error(f"Custom Eval - Validation image directory not found: '{val_images_dir}'. Skipping.")
            return
        if not os.path.isfile(val_annotations_path_for_custom_eval):
            fold_py_logger.error(f"Custom Eval - Validation annotation file not found: '{val_annotations_path_for_custom_eval}'. Skipping.")
            return

        dataset_val = sv.DetectionDataset.from_coco(
            images_directory_path=val_images_dir,
            annotations_path=val_annotations_path_for_custom_eval,
            min_image_area_percentage=0.0, # Load all images regardless of annotation area
            force_masks=False # Assuming object detection, not segmentation
        )

        effective_classes = dataset_val.classes # These are from the COCO 'categories'
        if not effective_classes and classes: # If dataset_val couldn't load, use ones passed from main script (already sorted by ID)
            effective_classes = classes
            fold_py_logger.info(f"Using class names passed from main script: {effective_classes}")
        elif not effective_classes and not classes:
            fold_py_logger.warning("Custom Eval - No class names from COCO 'categories' and none provided. Attempting to infer from GT annotation IDs.")
            all_gt_class_ids_list = [d.class_id for d in dataset_val.annotations.values() if d.class_id is not None and len(d.class_id) > 0]
            if all_gt_class_ids_list:
                all_gt_class_ids = np.concatenate(all_gt_class_ids_list)
                if len(all_gt_class_ids) > 0:
                    num_classes_inferred = int(np.max(all_gt_class_ids)) + 1
                    effective_classes = [f"class_{i}" for i in range(num_classes_inferred)]
                    fold_py_logger.info(f"Custom Eval - Inferred {num_classes_inferred} classes based on max GT ID: {effective_classes}")
                else:
                    effective_classes = ["class_0"] # Fallback
                    fold_py_logger.warning("Custom Eval - GT annotations have no class_ids, using generic 'class_0'.")
            else:
                effective_classes = ["class_0"] # Fallback
                fold_py_logger.warning("Custom Eval - No GT annotations with class_ids found, using generic 'class_0'.")

        if not effective_classes: # Final fallback
            fold_py_logger.error("Custom Eval - CRITICAL: No effective classes could be determined. mAP calculation will likely be meaningless. Using ['class_0'].")
            effective_classes = ["class_0"]


        targets_for_map = []
        predictions_for_map = []
        fold_py_logger.info(f"Custom Eval - Running FINAL predictions on {len(dataset_val.images)} validation images...")
        if not dataset_val.images:
            fold_py_logger.error("Custom Eval - No images found in the validation dataset after loading. Skipping prediction loop.")
            return

        # Assuming eval_model handles its own device (e.g., was moved to GPU if available)
        # If not, you might need: eval_model.model.to(device_for_eval)

        for image_name, image_data in tqdm(dataset_val.images.items(), desc="Custom Final Eval"):
            if not isinstance(image_data, np.ndarray):
                fold_py_logger.warning(f"Image data for '{image_name}' is not a numpy array (type: {type(image_data)}). Skipping.")
                # Append empty detections to keep lists aligned for mAP calculation
                targets_for_map.append(sv.Detections.empty())
                predictions_for_map.append(sv.Detections.empty())
                continue
            
            # It's good practice to ensure the model and data are on the same device for prediction.
            # However, RFDETRLarge's .predict() might handle this internally or expect image_data as numpy array.

            predicted_detections = eval_model.predict(image_data, threshold=0.001) # Use a low threshold for mAP calculation
            ground_truth_detections = dataset_val.annotations.get(image_name, sv.Detections.empty())
            
            targets_for_map.append(ground_truth_detections)
            predictions_for_map.append(predicted_detections)

        fold_py_logger.info("Custom Eval - Calculating FINAL mAP...")
        if not predictions_for_map: # Or if all are empty
            fold_py_logger.error("Custom Eval - No predictions were generated. Skipping mAP calculation.")
            return
        
        num_map_classes = len(effective_classes)
        if num_map_classes == 0: # Should have been caught by fallback, but defense
            fold_py_logger.error("Custom Eval - Number of classes for mAP is 0. This is an error. Using 1 to avoid crash.")
            num_map_classes = 1

        map_calculator = MeanAveragePrecision.from_detections(
            ground_truth_detections_batch=targets_for_map,
            prediction_detections_batch=predictions_for_map,
            num_classes=num_map_classes,
            iou_thresholds=np.linspace(0.5, 0.95, 10)
        )

        map_value = map_calculator.mean_average_precision
        map_50_value = map_calculator.per_iou_threshold_map.get(0.50, np.nan) # Use np.nan if key missing

        processed_map_results = {
            'map': map_value if map_value is not None else np.nan,
            'map_50': map_50_value,
        }
        fold_py_logger.info(f"Custom Eval - FINAL mAP results (raw from calculator): map={map_value}, map_50={map_50_value}")
        fold_py_logger.info(f"Custom Eval - FINAL mAP results (processed for CSV): {processed_map_results}")


        results_data = {'model_epoch': ['best']}
        map_keys_to_df_cols = {
            'map': 'metrics/mAP50-95(B)',
            'map_50': 'metrics/mAP50(B)',
        }
        for map_key, df_col_name in map_keys_to_df_cols.items():
            val = processed_map_results.get(map_key, np.nan)
            results_data[df_col_name] = [val if not np.isnan(val) else 0.0] # Store 0.0 for NaN in CSV for consistency

        df_results = pd.DataFrame(results_data)
        df_results_path = os.path.join(output_dir, "final_eval_results.csv")
        df_results.to_csv(df_results_path, index=False, float_format='%.5f')
        fold_py_logger.info(f"Custom Eval - Saved FINAL mAP results to {df_results_path}")

        cm_path = os.path.join(output_dir, "final_confusion_matrix.png")
        # Check if there are any non-empty ground truths or predictions
        has_gt_data = any(len(gt.xyxy) > 0 for gt in targets_for_map)
        has_pred_data = any(len(pred.xyxy) > 0 for pred in predictions_for_map)

        if has_gt_data and has_pred_data and effective_classes:
            try:
                cm = sv.ConfusionMatrix.from_detections(
                    targets=targets_for_map,
                    predictions=predictions_for_map,
                    classes=effective_classes
                )
                cm.plot(save_path=cm_path, class_names_rotation=45) # Updated for supervision's API if needed
                fold_py_logger.info(f"Custom Eval - Saved FINAL confusion matrix to {cm_path}")
            except ValueError as ve: # Catch specific errors like "Targets and predictions have different number of classes"
                fold_py_logger.error(f"Custom Eval - ValueError generating confusion matrix: {ve}. Ensure class consistency.", exc_info=True)
            except Exception as e_cm:
                fold_py_logger.error(f"Custom Eval - Error generating confusion matrix: {e_cm}", exc_info=True)
        elif not effective_classes:
            fold_py_logger.warning("Custom Eval - No class names for confusion matrix. Skipping CM plot.")
        else: # No GT or no Pred data
            fold_py_logger.warning("Custom Eval - No ground truth or prediction data with bounding boxes for confusion matrix. Skipping CM plot.")

    except Exception as e_eval:
        fold_py_logger.error(f"Error during FINAL custom evaluation setup or main loop: {e_eval}", exc_info=True)


def train_single_fold(fold_idx, target_gpu_id_for_this_fold):
    fold_number_display = fold_idx + 1
    logger_name = f"Fold-{fold_number_display}"

    fold_overall_results_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{fold_number_display}")
    os.makedirs(fold_overall_results_dir, exist_ok=True)
    rfdetr_native_output_dir = os.path.join(fold_overall_results_dir, "rfdetr_training_output")
    os.makedirs(rfdetr_native_output_dir, exist_ok=True)
    custom_eval_output_dir = os.path.join(fold_overall_results_dir, "standardized_evaluation_results")
    os.makedirs(custom_eval_output_dir, exist_ok=True)

    fold_py_logger = logging.getLogger(logger_name)
    fold_py_logger.propagate = False # Prevent duplicate logs to main handler
    fold_py_logger.setLevel(logging.INFO)
    if fold_py_logger.hasHandlers(): # Clear existing handlers for this logger if any (e.g., from previous failed run in same session)
        for handler in list(fold_py_logger.handlers):
            fold_py_logger.removeHandler(handler); handler.close()

    py_log_filename = f"script_log_fold_{fold_number_display}_gpu_{target_gpu_id_for_this_fold if target_gpu_id_for_this_fold is not None else 'cpu'}.log"
    py_log_file = os.path.join(fold_overall_results_dir, py_log_filename)
    file_handler = logging.FileHandler(py_log_file, mode='w') # 'w' to overwrite for each fold run
    formatter = logging.Formatter(f'%(asctime)s - {logger_name} - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    fold_py_logger.addHandler(file_handler)
    fold_py_logger.info(f"Initializing training for Fold {fold_number_display} on effective target: {'GPU ' + str(target_gpu_id_for_this_fold) if target_gpu_id_for_this_fold is not None else 'CPU'}.")

    dataset_root_dir_for_fold = BASE_DATA_PATH_TEMPLATE.format(fold_number_display)

    fold_py_logger.info("--- Starting Annotation Validation ---")
    train_ann_path_for_rfdetr = os.path.join(dataset_root_dir_for_fold, "train", "_annotations.coco.json")
    valid_ann_path_for_rfdetr = os.path.join(dataset_root_dir_for_fold, "valid", "_annotations.coco.json")

    train_anns_have_issues = validate_coco_annotations(train_ann_path_for_rfdetr, fold_py_logger)
    valid_anns_have_issues = validate_coco_annotations(valid_ann_path_for_rfdetr, fold_py_logger)

    if train_anns_have_issues or valid_anns_have_issues:
        fold_py_logger.critical(f"Critical issues found in annotation files for Fold {fold_number_display}. Skipping training. Please check logs and fix data.")
        for handler_to_close in list(fold_py_logger.handlers): # Clean up this fold's logger
            handler_to_close.close()
            fold_py_logger.removeHandler(handler_to_close)
        return
    fold_py_logger.info("--- Annotation Validation Passed (basic structural checks completed) ---")

    ### --- MODIFICATION START --- ###
    # 2. DETERMINE NUMBER OF CLASSES AND MODIFY MODEL INITIALIZATION

    # Get the number of classes directly from the training annotations file.
    num_classes = get_num_classes_from_coco(train_ann_path_for_rfdetr)
    fold_py_logger.info(f"Determined the model needs to support {num_classes} classes from '{train_ann_path_for_rfdetr}'.")

    if num_classes == 0:
        fold_py_logger.critical(f"Could not determine the number of classes from the annotation file. Found 0 categories. Skipping fold {fold_number_display}.")
        for handler_to_close in list(fold_py_logger.handlers): # Clean up
            handler_to_close.close()
            fold_py_logger.removeHandler(handler_to_close)
        return
    ### --- MODIFICATION END --- ###


    actual_val_images_dir = os.path.join(dataset_root_dir_for_fold, "valid") # RFDETR uses 'valid'
    path_val_ann_for_custom_eval = valid_ann_path_for_rfdetr

    fold_py_logger.info(f"Base data directory for RFDETR for this fold: {dataset_root_dir_for_fold}")
    fold_py_logger.info(f"Custom eval will use VALIDATION images from: {actual_val_images_dir}")
    fold_py_logger.info(f"Custom eval will use VALIDATION annotations from: {path_val_ann_for_custom_eval}")

    if not os.path.isdir(dataset_root_dir_for_fold):
        fold_py_logger.error(f"CRITICAL - Base data directory for RFDETR not found: {dataset_root_dir_for_fold}. Cannot proceed.")
        return
    if not os.path.isdir(os.path.join(dataset_root_dir_for_fold, "train")): # Check train image folder exists
        fold_py_logger.error(f"CRITICAL - Training image directory not found: {os.path.join(dataset_root_dir_for_fold, 'train')}. Check data structure.")
        return
    if not os.path.isdir(actual_val_images_dir): # Check valid image folder exists
        fold_py_logger.error(f"CRITICAL - Validation image directory not found: {actual_val_images_dir}. Check data structure.")
        return


    orig_stdout_main_process = sys.stdout
    orig_stderr_main_process = sys.stderr
    rfdetr_lib_log_filename = f"rfdetr_library_output_fold_{fold_number_display}_gpu_{target_gpu_id_for_this_fold if target_gpu_id_for_this_fold is not None else 'cpu'}.log"
    stdout_stderr_log_file = os.path.join(fold_overall_results_dir, rfdetr_lib_log_filename)

    f_log_rfdetr_file = None
    training_successful = False
    original_env_vars_state = {} # To store original state of env vars we modify
    env_vars_to_manage_for_fold = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES']

    try:
        f_log_rfdetr_file = open(stdout_stderr_log_file, 'w')
        sys.stdout = f_log_rfdetr_file # Redirect RFDETR's C-level prints
        sys.stderr = f_log_rfdetr_file
        fold_py_logger.info(f"RFDETR library's native stdout/stderr redirected to: {stdout_stderr_log_file}")
        print(f"--- This is the start of redirected stdout/stderr for Fold {fold_number_display} ---", flush=True)

        # Backup environment variables that might be changed
        for var_name in env_vars_to_manage_for_fold:
            original_env_vars_state[var_name] = os.environ.get(var_name) # Stores None if not set

        # --- Device and Environment Setup for RFDETRLarge ---
        device_param_for_rfdetr_train = 'cpu' # Default for .train(device=...)
        expected_model_init_device_type = 'cpu' # What RFDETRLarge.__init__ is likely to use

        if target_gpu_id_for_this_fold is not None and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id_for_this_fold)
            fold_py_logger.info(f"Set CUDA_VISIBLE_DEVICES='{target_gpu_id_for_this_fold}' for RFDETR context.")
            print(f"Script: Set CUDA_VISIBLE_DEVICES='{target_gpu_id_for_this_fold}'", flush=True)

            # DDP-like vars often checked by models with DDP support, even for single GPU.
            # RFDETRLarge's __init__ or underlying mechanisms might use these.
            master_port_for_fold = 29500 + fold_idx # Ensure unique port
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(master_port_for_fold)
            os.environ['RANK'] = '0'; os.environ['WORLD_SIZE'] = '1'; os.environ['LOCAL_RANK'] = '0'
            fold_py_logger.info(f"Set DDP-like env vars: MASTER_ADDR=localhost, PORT={master_port_for_fold}, RANK=0, WORLD_SIZE=1, LOCAL_RANK=0")
            print(f"Script: Set DDP-like env vars: MASTER_ADDR=localhost, PORT={master_port_for_fold}, RANK=0, WORLD_SIZE=1, LOCAL_RANK=0", flush=True)

            torch.cuda.set_device(0) # After CUDA_VISIBLE_DEVICES, PyTorch sees this as 'cuda:0'
            device_param_for_rfdetr_train = 'cuda:0'
            expected_model_init_device_type = 'cuda' # RFDETR's __init__ will likely try 'cuda'
            fold_py_logger.info(f"PyTorch active device context set to: cuda:0 (Logical, maps to physical GPU {target_gpu_id_for_this_fold}).")
            print(f"Script: PyTorch active device context: cuda:0 (Physical GPU {target_gpu_id_for_this_fold})", flush=True)
        else: # CPU case
            fold_py_logger.info("Target GPU not specified, CUDA not available, or DEBUG_FORCE_CPU is True. RFDETR will use CPU.")
            print("Script: Target GPU not specified/available or DEBUG_FORCE_CPU. RFDETR will use CPU.", flush=True)
            # Ensure CUDA_VISIBLE_DEVICES is not set, so PyTorch/RFDETR doesn't try to use a GPU it shouldn't.
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
                fold_py_logger.info("Unset CUDA_VISIBLE_DEVICES for CPU operation.")
                print("Script: Unset CUDA_VISIBLE_DEVICES for CPU operation.", flush=True)

        fold_py_logger.info(f"RFDETRLarge.__init__ will likely attempt to initialize model on device type: '{expected_model_init_device_type}'.")
        fold_py_logger.info(f"The .train() method will later be called with device parameter: '{device_param_for_rfdetr_train}'.")
        fold_py_logger.warning("USER ACTION REQUIRED: Re-check the 'max category_id' logged by the annotation validator. "
                               "If this ID is out of bounds for the number of classes your RFDETRLarge model expects "
                               "(especially if pre-trained), it's a VERY COMMON cause of CUDA asserts during model "
                               "initialization (e.g., when creating embedding layers or classification heads).")
        fold_py_logger.info("DEBUG TIP: If your RFDETRLarge library allows it (e.g., via an argument like "
                               "RFDETRLarge(checkpoint_path=None) or RFDETRLarge(load_pretrained_weights=False)), "
                               "try initializing the model *without* its pre-trained weights. If the error disappears, "
                               "the issue is likely with the checkpoint file itself (corruption, incompatibility).")

        
        ### --- MODIFICATION START --- ###
        # 3. PASS THE `num_classes` VARIABLE TO THE MODEL CONSTRUCTOR
        try:
            # The argument name is likely 'num_classes'. Check your library's source or docs if this fails.
            model_rf = RFDETRLarge(num_classes=num_classes)
            fold_py_logger.info(f"Successfully called RFDETRLarge() constructor with num_classes={num_classes}.")
        except TypeError as e:
            fold_py_logger.error(f"CRITICAL: Failed to initialize RFDETRLarge with 'num_classes' argument. The library might use a different argument name (e.g., 'nc'). Please check the library's documentation.", exc_info=True)
            raise e # Re-raise the exception to stop the script
        ### --- MODIFICATION END --- ###


        try:
            # Let's try to find out what device the model's parameters are actually on.
            # This assumes model_rf.model exists and has parameters.
            actual_model_device = next(model_rf.model.parameters()).device
            fold_py_logger.info(f"{MODEL_DESCRIPTOR} parameters are on device: {actual_model_device} after __init__.")
            if expected_model_init_device_type != actual_model_device.type:
                fold_py_logger.warning(f"WARNING: Expected model init device type '{expected_model_init_device_type}' but parameters are on '{actual_model_device.type}'. This might be an issue if devices are incompatible (e.g., CPU vs GPU).")
        except StopIteration: # Model might have no parameters yet, or structure is different
            fold_py_logger.warning(f"Could not determine actual model device after instantiation (model may have no parameters yet or a different structure).")
        except AttributeError: # model_rf.model might not exist
            fold_py_logger.warning(f"Could not access model_rf.model.parameters() to determine device. Structure might be different.")
        except Exception as e_get_device:
            fold_py_logger.warning(f"Could not determine actual model device after instantiation due to an unexpected error: {e_get_device}")


        current_train_params = RFDETR_TRAIN_PARAMS.copy()
        current_train_params['dataset_dir'] = dataset_root_dir_for_fold
        current_train_params['output_dir'] = rfdetr_native_output_dir
        current_train_params['device'] = device_param_for_rfdetr_train # Pass the desired device for the .train() method

        fold_py_logger.info(f"Starting RFDETR native training with parameters for its .train() method: {current_train_params}")
        print(f"Script: Starting RFDETR native training with parameters for .train(): {current_train_params}", flush=True)

        model_rf.train(**current_train_params)

        training_successful = True
        fold_py_logger.info("RFDETR native training completed.")
        print(f"Script: RFDETR native training completed. Outputs expected in {rfdetr_native_output_dir}", flush=True)

    except Exception as e_train_or_init: # Catch errors from RFDETRLarge() or model_rf.train()
        training_successful = False
        fold_py_logger.error(f"ERROR during RFDETR instantiation or training for Fold {fold_number_display}: {e_train_or_init}", exc_info=True)
        # The full traceback will be in the fold_py_logger file.
        # The original error message from RFDETR's internal print (if any) would have gone to the redirected stderr.
        try:
            if sys.stderr and not sys.stderr.closed: # sys.stderr is f_log_rfdetr_file here
                # This message will go into the rfdetr_library_output_...log
                print(f"\nCRITICAL SCRIPT-LEVEL ERROR ENCOUNTERED (see details above in this log or in the separate script_log_fold_...log):\n{e_train_or_init}\n", file=sys.stderr, flush=True)
                # Also print traceback to the redirected stderr for convenience
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
        except Exception as e_print_err_to_redirect:
            # If printing to redirected stderr fails, print to original stderr
            print(f"ADDITIONAL ERROR: Could not print critical error to redirected stderr. Original error: {e_train_or_init}. Print error: {e_print_err_to_redirect}", file=orig_stderr_main_process, flush=True)

    finally:
        if f_log_rfdetr_file and not f_log_rfdetr_file.closed:
            try:
                print(f"\n--- End: RFDETR Lib Output Redirection for Fold {fold_number_display} ---\n", file=f_log_rfdetr_file, flush=True)
            except Exception:
                pass # Avoid error during cleanup
            f_log_rfdetr_file.close()

        # Restore original stdout/stderr
        sys.stdout = orig_stdout_main_process
        sys.stderr = orig_stderr_main_process
        fold_py_logger.info("Restored main process stdout/stderr.")

        # Restore original environment variables
        for var_name_restore, orig_val in original_env_vars_state.items():
            if orig_val is None: # If it wasn't set originally
                if var_name_restore in os.environ:
                    del os.environ[var_name_restore]
            else: # If it was set, restore it
                os.environ[var_name_restore] = orig_val
        fold_py_logger.info("Restored environment variables to their pre-fold state.")

        # Attempt to clean up distributed process group if RFDETR initialized one
        if target_gpu_id_for_this_fold is not None and torch.cuda.is_available(): # Only if GPU was targeted
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                fold_py_logger.info(f"Attempting to destroy distributed process group potentially initialized by RFDETR for Fold {fold_number_display}.")
                try:
                    torch.distributed.destroy_process_group()
                    fold_py_logger.info("Distributed process group destroyed successfully.")
                except Exception as e_destroy:
                    fold_py_logger.error(f"Error destroying distributed process group: {e_destroy}", exc_info=True)
            else:
                fold_py_logger.info(f"PyTorch distributed module not initialized (or not available) at end of Fold {fold_number_display} processing, skipping destroy_process_group.")


    if training_successful:
        fold_py_logger.info("Training successful. Proceeding to parse metrics and perform custom evaluation.")
        parse_rfdetr_native_metrics(
            rfdetr_metrics_csv_path=os.path.join(rfdetr_native_output_dir, "metrics.csv"),
            comparable_csv_output_path=os.path.join(custom_eval_output_dir, "epoch_metrics_comparable.csv"),
            fold_py_logger=fold_py_logger
        )
        best_checkpoint_path = os.path.join(rfdetr_native_output_dir, "best.pth")
        if not os.path.exists(best_checkpoint_path):
            fold_py_logger.warning(f"'best.pth' not found in {rfdetr_native_output_dir}. Trying 'latest.pth'.")
            best_checkpoint_path = os.path.join(rfdetr_native_output_dir, "latest.pth")

        if os.path.exists(best_checkpoint_path):
            fold_py_logger.info(f"Loading model from checkpoint for custom evaluation: {best_checkpoint_path}")
            
            # Setup environment for evaluation model loading
            original_cvd_eval_fold_scope = os.environ.get('CUDA_VISIBLE_DEVICES') # Save current CVD
            eval_device_log_str = 'cpu'

            try:
                if target_gpu_id_for_this_fold is not None and torch.cuda.is_available():
                    # Ensure eval model loads on the same GPU used for training if GPU was used
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id_for_this_fold)
                    if torch.cuda.is_available(): # Re-check, just in case
                        torch.cuda.set_device(0) # Use logical device 0 of the visible devices
                    eval_device_log_str = f'cuda:0 (physical GPU {target_gpu_id_for_this_fold})'
                    fold_py_logger.info(f"Set CUDA_VISIBLE_DEVICES='{target_gpu_id_for_this_fold}' for eval model loading. PyTorch active: cuda:0.")
                else: # CPU for evaluation
                    if 'CUDA_VISIBLE_DEVICES' in os.environ: del os.environ['CUDA_VISIBLE_DEVICES']
                    fold_py_logger.info("Using CPU for loading and running evaluation model.")

                # Instantiate RFDETRLarge for evaluation
                # It will likely pick up device based on CUDA_VISIBLE_DEVICES and its internal logic
                eval_model = RFDETRLarge(checkpoint_path=best_checkpoint_path)
                fold_py_logger.info(f"Successfully loaded {MODEL_DESCRIPTOR} from {best_checkpoint_path} for evaluation.")
                
                try:
                    actual_eval_model_device = next(eval_model.model.parameters()).device
                    fold_py_logger.info(f"Evaluation model parameters are on device: {actual_eval_model_device}.")
                    # If on CPU and GPU was intended (or vice-versa), this might be an issue with RFDETRLarge's behavior
                    # For now, assume RFDETRLarge respects CUDA_VISIBLE_DEVICES or handles CPU correctly.
                except Exception:
                    fold_py_logger.warning(f"Could not determine eval model's device. Assuming it's correct based on {eval_device_log_str} context.")


                classes_from_coco = []
                if os.path.exists(path_val_ann_for_custom_eval):
                    try:
                        with open(path_val_ann_for_custom_eval, 'r') as f_coco:
                            coco_data = json.load(f_coco)
                        if 'categories' in coco_data and isinstance(coco_data['categories'], list) and coco_data['categories']:
                            # Sort by ID to ensure consistent class order for supervision metrics
                            sorted_categories = sorted(coco_data['categories'], key=lambda x: x.get('id', float('inf')))
                            classes_from_coco = [cat['name'] for cat in sorted_categories]
                            fold_py_logger.info(f"Extracted {len(classes_from_coco)} class names for custom evaluation: {classes_from_coco}")
                        else:
                            fold_py_logger.warning(f"No 'categories' list found, or it's empty in {path_val_ann_for_custom_eval}. Custom eval class names will be inferred or default.")
                    except Exception as e_coco:
                        fold_py_logger.error(f"Error reading class names from COCO JSON {path_val_ann_for_custom_eval}: {e_coco}", exc_info=True)
                else:
                    fold_py_logger.warning(f"Validation annotation file for custom eval not found at {path_val_ann_for_custom_eval}. Class names might be missing or inferred.")

                generate_evaluation_outputs(
                    eval_model=eval_model, val_images_dir=actual_val_images_dir,
                    val_annotations_path_for_custom_eval=path_val_ann_for_custom_eval,
                    classes=classes_from_coco, output_dir=custom_eval_output_dir,
                    fold_py_logger=fold_py_logger
                )
            except Exception as e_load_eval:
                fold_py_logger.error(f"Error during model loading for eval or custom evaluation execution: {e_load_eval}", exc_info=True)
            finally:
                # Restore original CUDA_VISIBLE_DEVICES for the main script context
                if target_gpu_id_for_this_fold is not None: # Only if we might have changed it
                    if original_cvd_eval_fold_scope is None: # If it wasn't set before this try block
                        if 'CUDA_VISIBLE_DEVICES' in os.environ: del os.environ['CUDA_VISIBLE_DEVICES']
                    else: # Restore to its previous state for this fold's scope
                        os.environ['CUDA_VISIBLE_DEVICES'] = original_cvd_eval_fold_scope
                fold_py_logger.info("Restored CUDA_VISIBLE_DEVICES after evaluation model handling (if it was changed for eval).")
        else:
            fold_py_logger.error(f"No checkpoint ('best.pth' or 'latest.pth') found in {rfdetr_native_output_dir}. Skipping custom evaluation.")
    else:
        fold_py_logger.error("Training was not successful for this fold. Skipping metrics parsing and custom evaluation.")

    # Close and remove handlers for this fold's logger to free up file lock
    for handler in list(fold_py_logger.handlers):
        handler.close()
        fold_py_logger.removeHandler(handler)
    logging.info(f"Fold {fold_number_display} processing finished. Log: {py_log_file}")


def run_cross_validation_training():
    logging.info(f"Starting {NUM_FOLDS}-fold Cross-Validation for {MODEL_DESCRIPTOR} (SEQUENTIAL EXECUTION).")
    logging.info(f"Main results will be saved in: {BASE_RESULTS_DIR}")
    logging.info(f"Base data path template: {BASE_DATA_PATH_TEMPLATE}")
    logging.info(f"Model: {MODEL_DESCRIPTOR}")
    logging.info(f"RFDETR Training Parameters (for .train() method): {RFDETR_TRAIN_PARAMS}")
    
    if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':
        logging.info("CUDA_LAUNCH_BLOCKING is set to 1 (helps with CUDA error reporting).")
    if os.environ.get('TORCH_USE_CUDA_DSA') == '1':
        logging.info("TORCH_USE_CUDA_DSA is set to 1 (attempts detailed device-side assertions).")
    else:
        logging.warning("TORCH_USE_CUDA_DSA is NOT set. For more detailed CUDA errors, consider setting "
                        "it in your shell ('export TORCH_USE_CUDA_DSA=1') before running this script.")

    effective_target_gpu_id_for_folds = None # Can be None (CPU) or an int (GPU ID)

    if DEBUG_FORCE_CPU:
        logging.warning("DEBUG_FORCE_CPU is True. All folds will attempt to run on CPU.")
        effective_target_gpu_id_for_folds = None
    elif torch.cuda.is_available():
        num_gpus_main_scope = torch.cuda.device_count()
        if num_gpus_main_scope > 0:
            if TARGET_GPU_ID_FOR_SEQUENTIAL_RUN >= num_gpus_main_scope:
                logging.warning(f"Requested GPU ID {TARGET_GPU_ID_FOR_SEQUENTIAL_RUN} is out of range "
                                f"(found {num_gpus_main_scope} GPUs). Defaulting to GPU 0.")
                effective_target_gpu_id_for_folds = 0
            else:
                effective_target_gpu_id_for_folds = TARGET_GPU_ID_FOR_SEQUENTIAL_RUN
            logging.info(f"All sequential folds will attempt to run on GPU: {effective_target_gpu_id_for_folds}")
        else: # CUDA available but no GPUs detected by PyTorch
            logging.warning("CUDA is reported as available by PyTorch, but no GPUs were detected (device_count is 0). "
                            "Training will proceed on CPU.")
            effective_target_gpu_id_for_folds = None
    else: # CUDA not available
        logging.warning("CUDA not available. Training will proceed on CPU.")
        effective_target_gpu_id_for_folds = None

    if NUM_FOLDS == 0:
        logging.info("NUM_FOLDS is set to 0. No training will be performed.")
        return

    for i in range(NUM_FOLDS):
        fold_idx_0_based = i
        fold_number_display = fold_idx_0_based + 1
        logging.info(f"\n{'='*25} Starting Fold {fold_number_display}/{NUM_FOLDS} {'='*25}")
        
        train_single_fold(fold_idx=fold_idx_0_based, target_gpu_id_for_this_fold=effective_target_gpu_id_for_folds)
        
        logging.info(f"{'='*25} Completed processing for Fold {fold_number_display}/{NUM_FOLDS} {'='*25}\n")
        
        if effective_target_gpu_id_for_folds is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logging.info(f"Cleared CUDA cache after Fold {fold_number_display}.")
            except RuntimeError as e_cache:
                logging.error(f"Error clearing CUDA cache after Fold {fold_number_display}: {e_cache}")


    logging.info(f"All {NUM_FOLDS}-fold Cross-Validation tasks have finished.")
    logging.info(f"Check the {BASE_RESULTS_DIR} directory for all outputs and logs.")

if __name__ == '__main__':
    script_start_time = pd.Timestamp.now()
    logging.info(f"Executing {MODEL_DESCRIPTOR} Sequential Cross-Validation Training Script at {script_start_time}.")
    run_cross_validation_training()
    script_end_time = pd.Timestamp.now()
    logging.info(f"Script execution finished at {script_end_time}. Total duration: {script_end_time - script_start_time}.")