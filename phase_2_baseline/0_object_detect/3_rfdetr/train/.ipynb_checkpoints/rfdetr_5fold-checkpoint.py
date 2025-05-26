import os
import logging
import sys
import multiprocessing
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json # Added for loading COCO json

# --- Imports for RFDETR and Evaluation ---
from rfdetr import RFDETRLarge
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from tqdm import tqdm
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Matplotlib was imported but not used directly in provided snippets for plotting
# from sklearn.metrics import precision_recall_curve, average_precision_score # Kept if we want to add more later

# --- General Configuration ---
NUM_FOLDS = 5
BASE_DATA_PATH_TEMPLATE = "/blue/hulcr/gmarais/PhD/IBBI_work/phase_1_data/2_object_detection_phase_2/coco/cv_iteration_{}"
BASE_RESULTS_DIR = "./results_rfdetr_large_detailed_eval"
MODEL_DESCRIPTOR = "RFDETRLarge"

RFDETR_TRAIN_PARAMS = {
    "epochs": 3,
    "batch_size": 1,
    "grad_accum_steps": 32,
    "lr": 1e-4,
}

# --- Main Script Logging Setup (Orchestrator Log) ---
main_orchestration_log_dir = os.path.join(BASE_RESULTS_DIR, "orchestration_logs")
os.makedirs(main_orchestration_log_dir, exist_ok=True)
root_logger_main = logging.getLogger()
if root_logger_main.name == 'root':
    if root_logger_main.hasHandlers():
        for handler in list(root_logger_main.handlers):
            root_logger_main.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(main_orchestration_log_dir, f"cv_orchestration_main_{MODEL_DESCRIPTOR.lower()}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Multiprocessing Start Method Configuration ---
try:
    if multiprocessing.get_start_method(allow_none=True) is None or multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    logging.info(f"MainProcess: Multiprocessing start method configured to 'spawn'. Current method: {multiprocessing.get_start_method()}")
except RuntimeError as e:
    logging.warning(f"MainProcess: Could not set multiprocessing start method to 'spawn': {e}. Current method: {multiprocessing.get_start_method(allow_none=True)}")


def parse_rfdetr_native_metrics(rfdetr_metrics_csv_path, comparable_csv_output_path, fold_py_logger):
    """
    Parses RFDETR's native metrics.csv and reformats it to be as comparable as
    possible to the Ultralytics results.csv format.
    """
    fold_py_logger.info(f"Attempting to parse RFDETR native metrics from: {rfdetr_metrics_csv_path}")
    if not os.path.exists(rfdetr_metrics_csv_path):
        fold_py_logger.warning(f"RFDETR native metrics.csv not found at {rfdetr_metrics_csv_path}. Skipping comparable epoch metrics generation.")
        return

    try:
        df_rfdetr = pd.read_csv(rfdetr_metrics_csv_path)
        df_comparable = pd.DataFrame()

        if 'epoch' in df_rfdetr.columns:
            df_comparable['epoch'] = df_rfdetr['epoch']
        else:
            fold_py_logger.warning("Column 'epoch' not found in RFDETR metrics.csv.")
            df_comparable['epoch'] = np.nan

        df_comparable['time'] = np.nan
        df_comparable['train/box_loss'] = df_rfdetr.get('train_loss_bbox', np.nan) + df_rfdetr.get('train_loss_giou', 0)
        df_comparable['train/cls_loss'] = df_rfdetr.get('train_loss_ce', np.nan)
        df_comparable['train/dfl_loss'] = np.nan
        df_comparable['metrics/precision(B)'] = np.nan
        df_comparable['metrics/recall(B)'] = np.nan
        df_comparable['metrics/mAP50(B)'] = df_rfdetr.get('val_map_50', np.nan)
        df_comparable['metrics/mAP50-95(B)'] = df_rfdetr.get('val_map', np.nan)
        df_comparable['val/box_loss'] = df_rfdetr.get('val_loss_bbox', np.nan) + df_rfdetr.get('val_loss_giou', 0)
        df_comparable['val/cls_loss'] = df_rfdetr.get('val_loss_ce', np.nan)
        df_comparable['val/dfl_loss'] = np.nan
        df_comparable['lr/pg0'] = df_rfdetr.get('lr', np.nan)
        df_comparable['lr/pg1'] = df_rfdetr.get('lr', np.nan)
        df_comparable['lr/pg2'] = df_rfdetr.get('lr', np.nan)

        ultralytics_column_order = [
            'epoch', 'time', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'lr/pg0', 'lr/pg1', 'lr/pg2'
        ]
        
        for col in ultralytics_column_order:
            if col not in df_comparable.columns:
                df_comparable[col] = np.nan
        
        df_comparable = df_comparable[ultralytics_column_order]
        df_comparable.to_csv(comparable_csv_output_path, index=False, float_format='%.5f')
        fold_py_logger.info(f"Saved comparable epoch metrics to: {comparable_csv_output_path}")

    except Exception as e_parse:
        fold_py_logger.error(f"Error parsing RFDETR native metrics.csv: {e_parse}", exc_info=True)

# --- MODIFIED function signature ---
def generate_evaluation_outputs(eval_model, val_images_dir, val_annotations_path, classes, output_dir, fold_py_logger):
    """
    Generates FINAL evaluation metrics and plots for the BEST trained RFDETR model.
    Focuses on mAP breakdown and Confusion Matrix.
    val_images_dir: Direct path to the validation images folder.
    val_annotations_path: Direct path to the validation COCO JSON annotation file.
    """
    fold_py_logger.info(f"Starting FINAL model custom evaluation in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        fold_py_logger.info(f"Using validation images from: {val_images_dir}")
        fold_py_logger.info(f"Using validation annotations from: {val_annotations_path}")

        if not (os.path.isdir(val_images_dir) and os.path.isfile(val_annotations_path)):
            fold_py_logger.error(f"Validation images dir '{val_images_dir}' or annotations file '{val_annotations_path}' not found or invalid. Skipping FINAL custom evaluation.")
            return

        dataset_val = sv.DetectionDataset.from_coco(
            images_directory_path=val_images_dir,
            annotations_path=val_annotations_path
        )
        
        effective_classes = dataset_val.classes
        if not effective_classes and classes: 
            effective_classes = classes
        elif not effective_classes and not classes: 
            fold_py_logger.warning("No class names found for evaluation. Using generic names.")
            all_gt_class_ids = np.concatenate([d.class_id for d in dataset_val.annotations.values() if len(d.class_id)>0] or [np.array([])])
            num_classes = (np.max(all_gt_class_ids) + 1) if len(all_gt_class_ids) > 0 else 1 
            effective_classes = [f"class_{i}" for i in range(int(num_classes))]

        targets_for_map = []
        predictions_for_map = []
        
        fold_py_logger.info("Running FINAL predictions on validation set for custom evaluation...")
        # dataset_val is an iterable of (image_path, sv.Detections)
        # sv.DetectionDataset.from_coco loads images into dataset_val.images as {path: np.ndarray}
        # and annotations into dataset_val.annotations as {path: sv.Detections}
        # Iterating directly over dataset_val is preferred if it's set up as an iterator of (image_path, ground_truth_detections)
        # However, the original code iterated `for image_path, ground_truth_detections in tqdm(dataset_val, ...)`
        # Let's ensure image_path corresponds to an actual file path for Image.open()
        
        # tqdm expects an iterable. dataset_val itself should be iterable if it's a list of tuples or similar.
        # If dataset_val.images is a dictionary {image_path: image_data}, we iterate its keys for paths.
        for image_path_key in tqdm(dataset_val.images.keys(), desc="Final Eval"):
            ground_truth_detections = dataset_val.annotations[image_path_key]
            image = Image.open(image_path_key) # image_path_key is the actual file path
            predicted_detections = eval_model.predict(image, threshold=0.001) 
            targets_for_map.append(ground_truth_detections)
            predictions_for_map.append(predicted_detections)
        
        fold_py_logger.info("Calculating FINAL mAP...")
        pred_annotations_dict = {}
        # Ensure keys for pred_annotations_dict match the keys in dataset_val.annotations (which are image paths)
        for i, img_path_key in enumerate(dataset_val.images.keys()): 
            pred_annotations_dict[img_path_key] = predictions_for_map[i]

        prediction_ds_for_map = sv.DetectionDataset(
            images=dataset_val.images, 
            annotations=pred_annotations_dict,
            classes=effective_classes 
        )
        map_calculator = MeanAveragePrecision.from_coco_detections(
            ground_truth_dataset = dataset_val,
            prediction_dataset = prediction_ds_for_map
        )
        map_results_dict = map_calculator.compute() 
        fold_py_logger.info(f"FINAL mAP results: {map_results_dict}")

        results_data = {
            'model_epoch': ['best'],
            'metrics/mAP50-95(B)': [map_results_dict.get('map', 0)],
            'metrics/mAP50(B)': [map_results_dict.get('map_50', 0)],
            'metrics/mAP75(B)': [map_results_dict.get('map_75', 0)],
            'metrics/mAP_small(B)': [map_results_dict.get('map_small', 0)],
            'metrics/mAP_medium(B)': [map_results_dict.get('map_medium', 0)],
            'metrics/mAP_large(B)': [map_results_dict.get('map_large', 0)],
        }
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(os.path.join(output_dir, "final_eval_results.csv"), index=False, float_format='%.5f')
        fold_py_logger.info(f"Saved FINAL mAP results to {os.path.join(output_dir, 'final_eval_results.csv')}")
        
        fold_py_logger.info("Generating FINAL Confusion Matrix...")
        cm = sv.ConfusionMatrix.from_detections(
            predictions=predictions_for_map, 
            targets=targets_for_map,    
            classes=effective_classes
        )
        cm.plot(save_path=os.path.join(output_dir, "final_confusion_matrix.png"), class_names_rotation=45)
        fold_py_logger.info(f"Saved FINAL confusion matrix to {os.path.join(output_dir, 'final_confusion_matrix.png')}")
        
        fold_py_logger.warning("PR_curve.png, P_curve.png, R_curve.png like Ultralytics require extensive custom plotting and are not generated by this function. Focus is on mAP and Confusion Matrix.")

    except Exception as e_eval:
        fold_py_logger.error(f"Error during FINAL custom evaluation: {e_eval}", exc_info=True)


def train_single_fold(fold_idx, assigned_gpu_id):
    fold_number_display = fold_idx + 1
    current_process_name = multiprocessing.current_process().name

    fold_overall_results_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{fold_number_display}")
    os.makedirs(fold_overall_results_dir, exist_ok=True)

    rfdetr_native_output_dir = os.path.join(fold_overall_results_dir, "rfdetr_training_output")
    os.makedirs(rfdetr_native_output_dir, exist_ok=True)
    
    custom_eval_output_dir = os.path.join(fold_overall_results_dir, "standardized_evaluation_results")
    os.makedirs(custom_eval_output_dir, exist_ok=True)

    fold_py_logger = logging.getLogger(current_process_name)
    fold_py_logger.propagate = False
    fold_py_logger.setLevel(logging.INFO)
    if fold_py_logger.hasHandlers():
        for handler in list(fold_py_logger.handlers):
            fold_py_logger.removeHandler(handler)
            handler.close()
    py_log_file = os.path.join(fold_overall_results_dir, f"script_log_fold_{fold_number_display}_gpu_{assigned_gpu_id}.log")
    file_handler = logging.FileHandler(py_log_file, mode='a') # Changed to 'a' for append
    formatter = logging.Formatter(f'%(asctime)s - {current_process_name} - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    fold_py_logger.addHandler(file_handler)
    fold_py_logger.info(f"Initializing training for Fold {fold_number_display} on GPU {assigned_gpu_id} with model {MODEL_DESCRIPTOR}.")

    dataset_root_dir_for_fold = BASE_DATA_PATH_TEMPLATE.format(fold_number_display)
    fold_py_logger.info(f"Dataset root for fold {fold_number_display}: {dataset_root_dir_for_fold}")

    # --- MODIFIED SECTION: Check for actual annotation file paths and names ---
    # 🖍️ IMPORTANT: Replace "YOUR_TRAIN_IMAGE_SUBFOLDER_NAME" and "YOUR_VALID_IMAGE_SUBFOLDER_NAME"
    # with the actual names of your image folders within each cv_iteration_N directory.
    # For example, if images are in cv_iteration_N/train_data/ and cv_iteration_N/valid_data/
    # then use "train_data" and "valid_data" respectively.
    # If they are directly in cv_iteration_N/train/ and cv_iteration_N/valid/, use "train" and "valid".
    _YOUR_TRAIN_IMAGE_SUBFOLDER_NAME = "train" # <<< 🖍️ REPLACE THIS !!! (e.g., "train_images", "train2017", "train")
    _YOUR_VALID_IMAGE_SUBFOLDER_NAME = "val" # <<< 🖍️ REPLACE THIS !!! (e.g., "val_images", "val2017", "valid")

    path_train_ann_actual = os.path.join(dataset_root_dir_for_fold, "annotations", "instances_train.json")
    path_valid_ann_actual = os.path.join(dataset_root_dir_for_fold, "annotations", "instances_val.json")
    
    # Also define paths to image folders, as RFDETR will need them implicitly and custom eval explicitly
    actual_train_images_dir = os.path.join(dataset_root_dir_for_fold, _YOUR_TRAIN_IMAGE_SUBFOLDER_NAME)
    actual_valid_images_dir = os.path.join(dataset_root_dir_for_fold, _YOUR_VALID_IMAGE_SUBFOLDER_NAME)
    
    fold_py_logger.info(f"Expecting TRAIN annotations at: {path_train_ann_actual}")
    fold_py_logger.info(f"Expecting TRAIN images in: {actual_train_images_dir}")
    fold_py_logger.info(f"Expecting VALID annotations (for custom eval) at: {path_valid_ann_actual}")
    fold_py_logger.info(f"Expecting VALID images (for custom eval) in: {actual_valid_images_dir}")

    if not os.path.exists(path_train_ann_actual):
        fold_py_logger.error(f"CRITICAL - COCO training annotations not found at {path_train_ann_actual}. Skipping fold.")
        return
    if not os.path.isdir(actual_train_images_dir): # Check if train image directory exists
        fold_py_logger.error(f"CRITICAL - Training image directory not found at {actual_train_images_dir}. Skipping fold.")
        return
    
    if not os.path.exists(path_valid_ann_actual):
        fold_py_logger.warning(f"VALIDATION annotations for custom eval not found at {path_valid_ann_actual}. Custom evaluation will likely fail.")
        # Depending on RFDETR, this might also affect training if it uses this val set.
    if not os.path.isdir(actual_valid_images_dir): # Check if valid image directory exists
        fold_py_logger.warning(f"VALIDATION image directory for custom eval not found at {actual_valid_images_dir}. Custom evaluation will likely fail.")
    # --- END OF MODIFIED SECTION ---

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    stdout_stderr_log_file = os.path.join(fold_overall_results_dir, f"rfdetr_library_output_fold_{fold_number_display}_gpu_{assigned_gpu_id}.log")

    training_successful = False
    try:
        with open(stdout_stderr_log_file, 'w') as f_log:
            sys.stdout = f_log; sys.stderr = f_log
            fold_py_logger.info("Python-level stdout/stderr redirected to: " + stdout_stderr_log_file)
            print(f"--- Start of redirected output for {current_process_name} using {MODEL_DESCRIPTOR} ---", flush=True)

            model_rf = RFDETRLarge() 
            fold_py_logger.info(f"Instantiated {MODEL_DESCRIPTOR} model.")
            print(f"RFDETR output: Instantiated {MODEL_DESCRIPTOR} model.", flush=True)

            device_to_use = str(assigned_gpu_id) if assigned_gpu_id is not None else 'cpu'
            if assigned_gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu_id)
                fold_py_logger.info(f"Set CUDA_VISIBLE_DEVICES={assigned_gpu_id}")

            current_train_params = RFDETR_TRAIN_PARAMS.copy()
            # 'dataset_dir' points to the root of the fold (e.g., .../cv_iteration_1/).
            # RFDETR's train() method is expected to locate 'annotations/instances_train.json',
            # 'annotations/instances_val.json' (if it uses one), and image folders
            # (e.g., 'YOUR_TRAIN_IMAGE_SUBFOLDER_NAME', 'YOUR_VALID_IMAGE_SUBFOLDER_NAME')
            # relative to this dataset_dir.
            current_train_params['dataset_dir'] = dataset_root_dir_for_fold 
            current_train_params['output_dir'] = rfdetr_native_output_dir 
            current_train_params['device'] = device_to_use

            fold_py_logger.info(f"Starting RFDETR training with parameters: {current_train_params}")
            print(f"RFDETR output: Starting training with parameters: {current_train_params}", flush=True)
            
            model_rf.train(**current_train_params) 
            training_successful = True

            fold_py_logger.info("RFDETR native training completed.")
            print(f"RFDETR output: Native training completed. Outputs in: {rfdetr_native_output_dir}", flush=True)

    except Exception as e_train:
        fold_py_logger.error(f"An error occurred during RFDETR training: {e_train}", exc_info=True)
    finally:
        sys.stdout = orig_stdout; sys.stderr = orig_stderr # Ensure restoration
        fold_py_logger.info("Restored stdout/stderr for this fold process.")

    if training_successful:
        rfdetr_metrics_file = os.path.join(rfdetr_native_output_dir, "metrics.csv")
        comparable_epoch_metrics_file = os.path.join(custom_eval_output_dir, "epoch_metrics_comparable.csv")
        parse_rfdetr_native_metrics(rfdetr_metrics_file, comparable_epoch_metrics_file, fold_py_logger)

        fold_py_logger.info("Proceeding to standardized custom evaluation of the BEST model...")
        best_checkpoint_path = os.path.join(rfdetr_native_output_dir, "best.pth")
        if not os.path.exists(best_checkpoint_path):
            fold_py_logger.warning(f"'best.pth' not found in {rfdetr_native_output_dir}. Attempting 'latest.pth'.")
            best_checkpoint_path = os.path.join(rfdetr_native_output_dir, "latest.pth")
        
        if os.path.exists(best_checkpoint_path):
            fold_py_logger.info(f"Loading checkpoint for final evaluation: {best_checkpoint_path}")
            eval_model = RFDETRLarge(checkpoint_path=best_checkpoint_path)
            
            # actual_valid_images_dir and path_valid_ann_actual are already defined above
            fold_py_logger.info(f"Custom evaluation will use validation images from: {actual_valid_images_dir}")
            fold_py_logger.info(f"Custom evaluation will use validation annotations from: {path_valid_ann_actual}")

            classes_from_coco = []
            if os.path.exists(path_valid_ann_actual): # Check if val ann file exists before trying to load
                try:
                    with open(path_valid_ann_actual, 'r') as f_coco: 
                        coco_data = json.load(f_coco)
                    classes_from_coco = [cat['name'] for cat in coco_data.get('categories', [])]
                    if not classes_from_coco: fold_py_logger.warning(f"Could not extract class names from COCO validation annotations: {path_valid_ann_actual}")
                except Exception as e_coco:
                    fold_py_logger.error(f"Error reading class names from COCO validation annotations ({path_valid_ann_actual}): {e_coco}")
            else:
                fold_py_logger.warning(f"Validation annotation file {path_valid_ann_actual} not found. Cannot load class names for custom evaluation.")

            # Pass the correct image and annotation paths to the evaluation function
            generate_evaluation_outputs(eval_model, 
                                        actual_valid_images_dir, 
                                        path_valid_ann_actual, 
                                        classes_from_coco, 
                                        custom_eval_output_dir, 
                                        fold_py_logger)
        else:
            fold_py_logger.error(f"No suitable checkpoint (best.pth or latest.pth) found for final custom evaluation. Skipping.")
    else:
        fold_py_logger.error("Training was not successful. Skipping metrics parsing and custom evaluation.")

    # Close file handlers for the fold-specific logger
    for handler in list(fold_py_logger.handlers): # Use list to avoid modifying during iteration
        handler.close()
        fold_py_logger.removeHandler(handler)


def run_cross_validation_training():
    logging.info(f"MainProcess: Starting {NUM_FOLDS}-fold CV for {MODEL_DESCRIPTOR} with custom evaluation...")
    logging.info(f"MainProcess: Base results directory set to: {BASE_RESULTS_DIR}")
    logging.info(f"MainProcess: COCO Data path template: {BASE_DATA_PATH_TEMPLATE}")
    logging.info(f"MainProcess: Using model: {MODEL_DESCRIPTOR}")
    logging.info(f"MainProcess: RFDETR Training Parameters: {RFDETR_TRAIN_PARAMS}")
    logging.warning(f"MainProcess: Using {MODEL_DESCRIPTOR}. This is a large model. Ensure RFDETR_TRAIN_PARAMS (especially batch_size) are configured for your available GPU memory.")
    logging.info("MainProcess: Ensure HF_TOKEN and ROBOFLOW_API_KEY are set in your environment if RFDETR requires them for any operations.")

    num_available_gpus = 0
    try:
        if torch.cuda.is_available():
            num_available_gpus = torch.cuda.device_count()
            logging.info(f"MainProcess: PyTorch detects {num_available_gpus} available CUDA GPU(s).")
        else:
            logging.warning("MainProcess: PyTorch reports CUDA not available. Training will attempt CPU if device is not set.")

        if num_available_gpus == 0 and NUM_FOLDS > 0 :
            logging.warning("MainProcess: No CUDA GPUs detected by PyTorch. Training will proceed on CPU if 'assigned_gpu_id' is None.")
    except Exception as e:
        logging.error(f"MainProcess: Could not verify GPU count using PyTorch: {e}. Ensure PyTorch is installed correctly with CUDA support.")
        return

    if NUM_FOLDS == 0:
        logging.info("MainProcess: NUM_FOLDS is 0. No training will be performed.")
        return

    if num_available_gpus == 0:
        logging.warning("MainProcess: No GPUs available. Assigning all folds to run on CPU (device will be 'cpu').")
        gpu_assignment_for_folds = [None] * NUM_FOLDS
    else:
        if num_available_gpus < NUM_FOLDS:
            logging.warning(
                f"MainProcess: Requested {NUM_FOLDS} folds, but only {num_available_gpus} GPU(s) are detected. "
                f"Folds will be assigned to available GPUs cyclically. This may lead to multiple processes trying to use the same GPU simultaneously if not managed carefully, potentially causing OOM errors, especially with {MODEL_DESCRIPTOR}."
            )
        gpu_assignment_for_folds = [i % num_available_gpus for i in range(NUM_FOLDS)]

    logging.info(f"MainProcess: GPU assignments for folds (Fold Index -> Assigned GPU ID or None for CPU):")
    for i in range(NUM_FOLDS):
        assigned_gpu_for_log = gpu_assignment_for_folds[i] if gpu_assignment_for_folds[i] is not None else 'CPU'
        logging.info(f"MainProcess:    Fold {i+1} (0-indexed: {i}) --> GPU {assigned_gpu_for_log}")

    training_processes = []
    for i in range(NUM_FOLDS):
        fold_idx_0_based = i
        assigned_gpu = gpu_assignment_for_folds[i]
        process_gpu_name_part = str(assigned_gpu) if assigned_gpu is not None else 'CPU'
        process_name = f"Fold-{fold_idx_0_based+1}_GPU-{process_gpu_name_part}_{MODEL_DESCRIPTOR}"

        logging.info(f"MainProcess: Preparing to launch training process {process_name} for Fold {fold_idx_0_based+1} on device '{process_gpu_name_part}'.")
        p = multiprocessing.Process(
            target=train_single_fold,
            args=(fold_idx_0_based, assigned_gpu),
            name=process_name
        )
        training_processes.append(p)
    
    logging.info(f"MainProcess: Starting {len(training_processes)} training processes...")
    for p in training_processes:
        p.start()
        logging.info(f"MainProcess: Process {p.name} (PID: {p.pid}) has been started.")

    logging.info("MainProcess: All training processes have been launched. Waiting for them to complete...")
    for i, p in enumerate(training_processes):
        p.join() 
        logging.info(f"MainProcess: Process {p.name} (PID: {p.pid}) for Fold {i+1} has completed. Exit code: {p.exitcode}.")
        if p.exitcode != 0:
            fold_results_project_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{i+1}") # Corrected variable name
            logging.error(f"MainProcess: Process {p.name} for Fold {i+1} (PID: {p.pid}) exited with a non-zero code ({p.exitcode}). "
                          f"Check its dedicated log files in {fold_results_project_dir} and any system error logs.")

    logging.info(f"MainProcess: All {NUM_FOLDS}-fold CV tasks for {MODEL_DESCRIPTOR} have finished processing.")
    logging.info(f"MainProcess: Check individual fold directories under {BASE_RESULTS_DIR} for native RFDETR outputs and standardized evaluation results.")


if __name__ == '__main__':
    logging.info(f"Starting {MODEL_DESCRIPTOR} CV training script with custom evaluation component.")
    run_cross_validation_training()