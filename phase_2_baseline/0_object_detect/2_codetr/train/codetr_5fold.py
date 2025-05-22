import os
import logging
import sys
import multiprocessing
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import subprocess
import shutil
import re 
import json
import pandas as pd
import numpy as np
import pickle 

import supervision as sv # For confusion matrix from .pkl

# --- General Configuration ---
NUM_FOLDS = 5
BASE_DATA_PATH_TEMPLATE = "/blue/hulcr/gmarais/PhD/phase_1_data/2_object_detection_phase_2/coco_formatted_data/cv_iteration_{}"
BASE_RESULTS_DIR = "./results_codetr_finetune" 
MODEL_DESCRIPTOR = "CoDETR_FineTuned"

# --- Co-DETR Specific Configuration (USER MUST UPDATE THESE) ---
CODETR_REPO_PATH = "/path/to/your/Co-DETR"  # Example: "/home/user/Co-DETR"

# **CRITICAL**: Update this to the config file of the "best performing" Co-DETR/Co-DINO model 
# you want to fine-tune (e.g., a Swin-L based Co-DINO model).
# This config should ideally have a 'load_from' field pointing to COCO pre-trained weights.
BASE_CODETR_CONFIG_FILE = os.path.join(CODETR_REPO_PATH, "projects/configs/co_dino/co_dino_5scale_swin_l_16xb1_1x_coco.py") # !! EXAMPLE !! VERIFY AND CHANGE
# Check your Co-DETR repo for available high-performance configs.

# **CRITICAL**: Set this to the number of classes in YOUR custom dataset.
NUM_CUSTOM_CLASSES = 20 # EXAMPLE: If your dataset has 20 classes. (COCO default is 80)

# --- Fine-tuning Hyperparameters for Co-DETR ---
CODETR_TRAIN_PARAMS = {
    "total_epochs": 24,       # Adjust for fine-tuning (MMDetection: runner.max_epochs)
    "samples_per_gpu": 1,     # Batch size per GPU (data.samples_per_gpu)
    "workers_per_gpu": 2,     # (data.workers_per_gpu)
    "base_lr": 2e-5,          # Often lower LR for fine-tuning (optimizer.lr)
    "checkpoint_interval": 1, # Save checkpoint every N epochs
    "evaluation_interval": 1, # Evaluate every N epochs
}

# --- Main Script Logging & Multiprocessing Setup (Same as before) ---
main_orchestration_log_dir = os.path.join(BASE_RESULTS_DIR, "orchestration_logs")
os.makedirs(main_orchestration_log_dir, exist_ok=True)
root_logger_main = logging.getLogger()
if root_logger_main.name == 'root':
    if root_logger_main.hasHandlers():
        for handler in list(root_logger_main.handlers): root_logger_main.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(os.path.join(main_orchestration_log_dir, f"cv_orchestration_main_{MODEL_DESCRIPTOR.lower().replace('-','_')}.log")),
                              logging.StreamHandler(sys.stdout)])
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn': multiprocessing.set_start_method('spawn', force=True)
    logging.info(f"MainProcess: Multiprocessing start method: {multiprocessing.get_start_method()}")
except RuntimeError as e:
    logging.warning(f"MainProcess: Could not set method to 'spawn': {e}. Current: {multiprocessing.get_start_method(allow_none=True)}")

# --- Helper Functions for Co-DETR ---

def modify_codetr_config(base_config_path, fold_config_path, data_root_for_fold, num_custom_classes, params, fold_py_logger):
    fold_py_logger.info(f"Modifying Co-DETR config '{base_config_path}' for fold, saving to '{fold_config_path}'")
    shutil.copyfile(base_config_path, fold_config_path)

    with open(fold_config_path, 'r') as f:
        config_content = f.read()

    # 1. Set data_root
    config_content = re.sub(r"data_root\s*=\s*'.*?'", f"data_root = r'{data_root_for_fold}/'", config_content, flags=re.DOTALL) # Use raw string for paths
    fold_py_logger.info(f"Set data_root = r'{data_root_for_fold}/'")

    # 2. Adjust dataset paths (train, val, test)
    # This assumes your custom data is in 'train/' and 'valid/' subdirs with COCO JSONs
    # For training dataset
    config_content = re.sub(r"(ann_file=data_root\s*\+\s*)'train/annotations/.*?\.json'", r"\1'train/_annotations.coco.json'", config_content)
    config_content = re.sub(r"(img_prefix=data_root\s*\+\s*)'train/.*?'", r"\1'train/images/'", config_content)
    config_content = re.sub(r"(data_prefix\s*=\s*dict\(img=data_root\s*\+\s*)'train/.*?'\)", r"\1'train/images/')", config_content)


    # For validation dataset (and usually test dataset points to the same for eval during training)
    config_content = re.sub(r"(ann_file=data_root\s*\+\s*)'val/annotations/.*?\.json'", r"\1'valid/_annotations.coco.json'", config_content)
    config_content = re.sub(r"(img_prefix=data_root\s*\+\s*)'val/.*?'", r"\1'valid/images/'", config_content)
    config_content = re.sub(r"(data_prefix\s*=\s*dict\(img=data_root\s*\+\s*)'val/.*?'\)", r"\1'valid/images/')", config_content)

    config_content = re.sub(r"(ann_file=data_root\s*\+\s*)'test/annotations/.*?\.json'", r"\1'valid/_annotations.coco.json'", config_content) # Point test to valid
    config_content = re.sub(r"(img_prefix=data_root\s*\+\s*)'test/.*?'", r"\1'valid/images/'", config_content)
    config_content = re.sub(r"(data_prefix\s*=\s*dict\(img=data_root\s*\+\s*)'test/.*?'\)", r"\1'valid/images/')", config_content)
    fold_py_logger.info("Attempted to update dataset paths for train, val, test to use 'train/' and 'valid/' subdirs.")

    # 3. Modify num_classes (CRITICAL for fine-tuning)
    # This pattern tries to find num_classes within model.bbox_head or model.roi_head.bbox_head
    # Co-DETR / Co-DINO often use model.bbox_head.num_classes
    config_content = re.sub(r"(bbox_head\s*=\s*dict\(.*?num_classes=)[\d]+", rf"\1{num_custom_classes}", config_content, flags=re.DOTALL)
    config_content = re.sub(r"(num_classes\s*=\s*)[\d]+(,\s*#\s*Number of classes)", rf"\1{num_custom_classes}\2", config_content) # If it's a simple line
    fold_py_logger.info(f"Set num_classes = {num_custom_classes}")

    # 4. Ensure 'load_from' is present if you want to fine-tune from COCO pre-trained weights.
    # The base config should ideally already have this. If not, you might need to add it.
    # Example: load_from = 'https://download.openmmlab.com/.../some_codino_swinl_model.pth'
    # This script assumes the base config handles 'load_from' correctly for COCO pretraining.

    # 5. Modify other training parameters from CODETR_TRAIN_PARAMS
    if "total_epochs" in params:
        config_content = re.sub(r"(runner\s*=\s*dict\(.*?max_epochs\s*=\s*)[\d]+", rf"\1{params['total_epochs']}", config_content, flags=re.DOTALL)
        config_content = re.sub(r"(max_epochs\s*=\s*)[\d]+", rf"\1{params['total_epochs']}", config_content, flags=re.DOTALL)
        fold_py_logger.info(f"Set max_epochs = {params['total_epochs']}")
    if "samples_per_gpu" in params:
        config_content = re.sub(r"(samples_per_gpu\s*=\s*)[\d]+", rf"\1{params['samples_per_gpu']}", config_content, flags=re.DOTALL)
        fold_py_logger.info(f"Set samples_per_gpu = {params['samples_per_gpu']}")
    if "workers_per_gpu" in params:
        config_content = re.sub(r"(workers_per_gpu\s*=\s*)[\d]+", rf"\1{params['workers_per_gpu']}", config_content, flags=re.DOTALL)
        fold_py_logger.info(f"Set workers_per_gpu = {params['workers_per_gpu']}")
    if "base_lr" in params:
        config_content = re.sub(r"(optimizer\s*=\s*dict\(.*?lr\s*=\s*)[\d\.e-]+", rf"\1{params['base_lr']}", config_content, flags=re.DOTALL)
        fold_py_logger.info(f"Set optimizer lr = {params['base_lr']}")
    if "checkpoint_interval" in params:
        config_content = re.sub(r"(checkpoint_config\s*=\s*dict\(interval\s*=\s*)[\d]+", rf"\1{params['checkpoint_interval']}", config_content, flags=re.DOTALL)
        fold_py_logger.info(f"Set checkpoint_config.interval = {params['checkpoint_interval']}")
    if "evaluation_interval" in params:
        config_content = re.sub(r"(evaluation\s*=\s*dict\(interval\s*=\s*)[\d]+", rf"\1{params['evaluation_interval']}", config_content, flags=re.DOTALL)
        fold_py_logger.info(f"Set evaluation.interval = {params['evaluation_interval']}")

    with open(fold_config_path, 'w') as f:
        f.write(config_content)
    fold_py_logger.info(f"Finished modifying config: {fold_config_path}. VERIFY IT MANUALLY.")


def parse_codetr_log_json(log_json_path, comparable_csv_output_path, fold_py_logger):
    # (This function remains largely the same as in the previous response)
    # Ensure keys like 'loss_cls', 'loss_bbox', 'loss_giou' match your Co-DETR model's log.json output
    fold_py_logger.info(f"Parsing Co-DETR .log.json from: {log_json_path}")
    if not os.path.exists(log_json_path):
        fold_py_logger.warning(f"Co-DETR .log.json not found. Skipping epoch metrics CSV generation for this fold.")
        return

    train_epoch_agg_losses = {} 
    val_epoch_data = []
    try:
        with open(log_json_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    epoch = log_entry.get("epoch")
                    mode = log_entry.get("mode")

                    if mode == "train" and epoch is not None:
                        if epoch not in train_epoch_agg_losses:
                            train_epoch_agg_losses[epoch] = {'loss_cls_sum': 0, 'loss_bbox_sum': 0, 
                                                             'loss_giou_sum': 0, 'loss_total_sum':0, # For overall 'loss' if available
                                                             'other_losses_sum':0, # For any other Co-DETR specific losses
                                                             'count': 0, 'lr_last': log_entry.get('lr', np.nan)}
                        
                        # Co-DETR / Co-DINO often have multiple detection heads and auxiliary losses.
                        # Sum relevant ones. This is an example, adjust based on your model's actual loss keys.
                        # Check your .log.json for keys like 'loss_cls_dn', 'loss_bbox_dn', 'loss_iou_dn' etc.
                        # For simplicity, we try to find common MMDetection loss patterns.
                        current_total_loss = 0
                        current_cls_loss = 0
                        current_bbox_loss = 0 # Sum of bbox, giou, iou etc.

                        for key, value in log_entry.items():
                            if 'loss_cls' in key: current_cls_loss += value
                            elif 'loss_bbox' in key: current_bbox_loss += value
                            elif 'loss_giou' in key: current_bbox_loss += value # Add GIOU to bbox loss
                            elif 'loss_iou' in key: current_bbox_loss += value # Add IOU to bbox loss
                            elif key == 'loss': current_total_loss = value # Overall loss
                        
                        train_epoch_agg_losses[epoch]['loss_cls_sum'] += current_cls_loss
                        train_epoch_agg_losses[epoch]['loss_bbox_sum'] += current_bbox_loss
                        train_epoch_agg_losses[epoch]['loss_total_sum'] += current_total_loss
                        train_epoch_agg_losses[epoch]['count'] += 1
                        train_epoch_agg_losses[epoch]['lr_last'] = log_entry.get('lr', train_epoch_agg_losses[epoch]['lr_last'])

                    elif mode == "val" and epoch is not None and "bbox_mAP" in log_entry: # Ensure it's a validation summary line
                        val_data = {
                            'epoch': epoch, 'time': log_entry.get("time"), 
                            'metrics/mAP50(B)': log_entry.get("bbox_mAP_50"),
                            'metrics/mAP50-95(B)': log_entry.get("bbox_mAP"),
                            'metrics/mAP75(B)': log_entry.get("bbox_mAP_75"),
                            'metrics/precision(B)': np.nan, 'metrics/recall(B)': np.nan,
                            'val/box_loss': np.nan, 'val/cls_loss': np.nan, 'val/dfl_loss': np.nan,
                        }
                        val_epoch_data.append(val_data)
                except json.JSONDecodeError: pass
        
        if not val_epoch_data: fold_py_logger.warning(f"No validation epoch data with 'bbox_mAP' found."); return
        df_comparable = pd.DataFrame(val_epoch_data)

        for epoch, train_stats in train_epoch_agg_losses.items():
            if train_stats['count'] > 0:
                idx = df_comparable[df_comparable['epoch'] == epoch].index
                if not idx.empty:
                    df_comparable.loc[idx, 'train/box_loss'] = train_stats['loss_bbox_sum'] / train_stats['count']
                    df_comparable.loc[idx, 'train/cls_loss'] = train_stats['loss_cls_sum'] / train_stats['count']
                    df_comparable.loc[idx, 'lr/pg0'] = train_stats['lr_last']
                    df_comparable.loc[idx, 'lr/pg1'] = train_stats['lr_last']
                    df_comparable.loc[idx, 'lr/pg2'] = train_stats['lr_last']
        
        df_comparable['train/dfl_loss'] = np.nan 
        ultralytics_column_order = ['epoch', 'time', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                                    'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                                    'metrics/mAP50-95(B)', 'metrics/mAP75(B)',
                                    'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                                    'lr/pg0', 'lr/pg1', 'lr/pg2']
        for col in ultralytics_column_order:
            if col not in df_comparable.columns: df_comparable[col] = np.nan
        df_comparable = df_comparable.sort_values(by='epoch')[ultralytics_column_order]
        df_comparable.to_csv(comparable_csv_output_path, index=False, float_format='%.5f')
        fold_py_logger.info(f"Saved comparable epoch metrics to: {comparable_csv_output_path}")
    except Exception as e_parse:
        fold_py_logger.error(f"Error parsing Co-DETR .log.json: {e_parse}", exc_info=True)


def run_codetr_final_evaluation(fold_config_path, checkpoint_path, custom_eval_output_dir, fold_py_logger, assigned_gpu_id_str, val_dataset_dir_for_fold, classes_from_coco):
    # (This function remains largely the same as in the previous response for Co-DETR)
    # Key parts: call tools/test.py, parse its mAP output, and attempt to generate confusion matrix from saved .pkl
    fold_py_logger.info(f"Running Co-DETR FINAL evaluation for checkpoint: {checkpoint_path}")
    os.makedirs(custom_eval_output_dir, exist_ok=True)
    
    test_script_path = os.path.join(CODETR_REPO_PATH, "tools/test.py")
    predictions_pkl_path = os.path.join(custom_eval_output_dir, "final_predictions.pkl")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = assigned_gpu_id_str
    # Update PYTHONPATH if necessary for your Co-DETR environment
    # env["PYTHONPATH"] = f"{CODETR_REPO_PATH}:{os.path.join(CODETR_REPO_PATH, 'dependencies/mmdetection')}:{env.get('PYTHONPATH', '')}"


    cmd_eval = ["python", test_script_path, fold_config_path, checkpoint_path, "--eval", "bbox", "--out", predictions_pkl_path]
    fold_py_logger.info(f"Executing Co-DETR test command: {' '.join(cmd_eval)}")
    codetr_eval_log_file = os.path.join(custom_eval_output_dir, "codetr_final_eval_stdout.log")

    try:
        with open(codetr_eval_log_file, 'w') as f_log:
            process = subprocess.run(cmd_eval, cwd=CODETR_REPO_PATH, env=env,
                                     stdout=f_log, stderr=subprocess.STDOUT, check=False, timeout=3600) # 1 hour timeout

        if process.returncode == 0:
            fold_py_logger.info(f"Co-DETR test.py completed. Log: {codetr_eval_log_file}")
            parsed_metrics = {}
            with open(codetr_eval_log_file, 'r') as f_log_read:
                log_content = f_log_read.read()
                map_overall_match = re.search(r"bbox_mAP\s*:\s*([\d\.]+)", log_content) # mAP @ 0.5:0.95
                map50_match = re.search(r"bbox_mAP_50\s*:\s*([\d\.]+)", log_content)
                map75_match = re.search(r"bbox_mAP_75\s*:\s*([\d\.]+)", log_content)
                if map_overall_match: parsed_metrics['metrics/mAP50-95(B)'] = float(map_overall_match.group(1))
                if map50_match: parsed_metrics['metrics/mAP50(B)'] = float(map50_match.group(1))
                if map75_match: parsed_metrics['metrics/mAP75(B)'] = float(map75_match.group(1))
            
            if parsed_metrics:
                df_final_metrics = pd.DataFrame([parsed_metrics])
                # Ensure standard columns for consistency, even if some are NaN
                standard_cols = ['metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/mAP75(B)', 
                                 'metrics/mAP_small(B)', 'metrics/mAP_medium(B)', 'metrics/mAP_large(B)']
                for col in standard_cols:
                    if col not in df_final_metrics.columns: df_final_metrics[col] = np.nan
                df_final_metrics = df_final_metrics[standard_cols]
                df_final_metrics.to_csv(os.path.join(custom_eval_output_dir, "final_eval_codetr_metrics.csv"), index=False, float_format='%.5f')
                fold_py_logger.info(f"Saved parsed Co-DETR final metrics.")
            else:
                fold_py_logger.warning("Could not parse mAP metrics from Co-DETR test.py output. Check log.")

            if os.path.exists(predictions_pkl_path):
                fold_py_logger.info(f"Attempting to generate Confusion Matrix from {predictions_pkl_path}")
                try:
                    with open(predictions_pkl_path, 'rb') as f_pkl: mmdet_predictions = pickle.load(f_pkl)
                    
                    # Load GT for CM
                    val_images_dir = os.path.join(val_dataset_dir_for_fold, "images")
                    val_ann_path = os.path.join(val_dataset_dir_for_fold, "_annotations.coco.json")
                    dataset_gt_for_cm = sv.DetectionDataset.from_coco(images_directory_path=val_images_dir, annotations_path=val_ann_path)
                    
                    effective_classes_for_cm = classes_from_coco if classes_from_coco else dataset_gt_for_cm.classes
                    if not effective_classes_for_cm: # Fallback if still no classes
                        num_cls_inferred = max(max(d.class_id)+1 for d in dataset_gt_for_cm.annotations.values() if d.class_id.size > 0) if any(d.class_id.size > 0 for d in dataset_gt_for_cm.annotations.values()) else NUM_CUSTOM_CLASSES
                        effective_classes_for_cm = [f"class_{i}" for i in range(int(num_cls_inferred))]


                    predictions_sv_list = []
                    targets_sv_list = []

                    if len(mmdet_predictions) == len(dataset_gt_for_cm.images):
                        for i, image_path_gt in enumerate(dataset_gt_for_cm.images): # Iterate in GT order
                            targets_sv_list.append(dataset_gt_for_cm.annotations[image_path_gt])
                            
                            xyxy_accum, conf_accum, cid_accum = [], [], []
                            # mmdet_predictions[i] is a list of arrays, one per class
                            for class_id, class_preds_np_array in enumerate(mmdet_predictions[i]):
                                if class_preds_np_array.shape[0] > 0:
                                    xyxy_accum.append(class_preds_np_array[:, :4])
                                    conf_accum.append(class_preds_np_array[:, 4])
                                    cid_accum.append(np.full(class_preds_np_array.shape[0], class_id))
                            
                            if xyxy_accum:
                                predictions_sv_list.append(sv.Detections(xyxy=np.concatenate(xyxy_accum),
                                                                         confidence=np.concatenate(conf_accum),
                                                                         class_id=np.concatenate(cid_accum)))
                            else:
                                predictions_sv_list.append(sv.Detections.empty())
                        
                        cm = sv.ConfusionMatrix.from_detections(predictions=predictions_sv_list, targets=targets_sv_list, classes=effective_classes_for_cm)
                        cm.plot(save_path=os.path.join(custom_eval_output_dir, "final_confusion_matrix.png"), class_names_rotation=45)
                        fold_py_logger.info(f"Saved final confusion matrix.")
                    else:
                        fold_py_logger.warning(f"Image count mismatch for CM: GT {len(dataset_gt_for_cm.images)}, Pred PKL {len(mmdet_predictions)}.")
                except Exception as e_cm: fold_py_logger.error(f"Error generating CM from .pkl: {e_cm}", exc_info=True)
            else: fold_py_logger.warning(f"Predictions .pkl not found. Skipping CM generation.")
        else: fold_py_logger.error(f"Co-DETR test.py failed. Code: {process.returncode}. Log: {codetr_eval_log_file}")
    except subprocess.TimeoutExpired: fold_py_logger.error(f"Co-DETR test.py timed out.")
    except Exception as e_eval_final: fold_py_logger.error(f"Exception in Co-DETR final eval: {e_eval_final}", exc_info=True)


def train_single_fold(fold_idx, assigned_gpu_id):
    # (Setup loggers, paths as before)
    fold_number_display = fold_idx + 1
    current_process_name = multiprocessing.current_process().name
    assigned_gpu_id_str = str(assigned_gpu_id)

    fold_overall_results_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{fold_number_display}")
    os.makedirs(fold_overall_results_dir, exist_ok=True)
    codetr_native_output_dir = os.path.join(fold_overall_results_dir, "codetr_training_output") 
    custom_eval_output_dir = os.path.join(fold_overall_results_dir, "standardized_evaluation_results")
    os.makedirs(custom_eval_output_dir, exist_ok=True)

    fold_py_logger = logging.getLogger(current_process_name)
    fold_py_logger.propagate = False; fold_py_logger.setLevel(logging.INFO)
    if fold_py_logger.hasHandlers():
        for handler in list(fold_py_logger.handlers): fold_py_logger.removeHandler(handler); handler.close()
    py_log_file = os.path.join(fold_overall_results_dir, f"script_log_fold_{fold_number_display}_gpu_{assigned_gpu_id_str}.log")
    file_handler = logging.FileHandler(py_log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter); fold_py_logger.addHandler(file_handler)
    fold_py_logger.info(f"STARTING Co-DETR Fine-tuning: Fold {fold_number_display} on GPU {assigned_gpu_id_str}")

    dataset_root_dir_for_fold = BASE_DATA_PATH_TEMPLATE.format(fold_number_display)
    fold_config_name = f"fold_{fold_number_display}_config_{MODEL_DESCRIPTOR.replace('-','_')}.py"
    fold_config_path = os.path.join(fold_overall_results_dir, fold_config_name)

    try:
        modify_codetr_config(BASE_CODETR_CONFIG_FILE, fold_config_path, dataset_root_dir_for_fold, NUM_CUSTOM_CLASSES, CODETR_TRAIN_PARAMS, fold_py_logger)
    except Exception as e_conf:
        fold_py_logger.error(f"FATAL: Failed to modify Co-DETR config: {e_conf}", exc_info=True); return

    codetr_train_stdout_log = os.path.join(codetr_native_output_dir, f"codetr_train_stdout_fold_{fold_number_display}.log")
    os.makedirs(codetr_native_output_dir, exist_ok=True) # Ensure work_dir exists for logging
    
    training_successful = False
    train_script_path = os.path.join(CODETR_REPO_PATH, "tools/train.py")
    cmd_train = ["python", "-u", train_script_path, fold_config_path, "--work-dir", codetr_native_output_dir, "--launcher", "none"]
    
    fold_py_logger.info(f"Executing Co-DETR train: {' '.join(cmd_train)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = assigned_gpu_id_str
    # CRITICAL: Ensure PYTHONPATH is correctly set if Co-DETR/MMDet is not installed in the environment globally
    # Example: env["PYTHONPATH"] = f"{CODETR_REPO_PATH}:{os.path.join(CODETR_REPO_PATH, 'mmdetection_path_if_forked')}:{env.get('PYTHONPATH', '')}"
    # If Co-DETR was installed with `pip install -e .` in its root, this might be less critical.

    try:
        with open(codetr_train_stdout_log, 'w') as f_log:
            process = subprocess.run(cmd_train, cwd=CODETR_REPO_PATH, env=env,
                                     stdout=f_log, stderr=subprocess.STDOUT, check=False, timeout=72000) # 20h timeout
        if process.returncode == 0: training_successful = True
        else: fold_py_logger.error(f"Co-DETR train failed. Code: {process.returncode}. See log: {codetr_train_stdout_log}")
    except subprocess.TimeoutExpired: fold_py_logger.error(f"Co-DETR training timed out. Log: {codetr_train_stdout_log}")
    except Exception as e_train_run: fold_py_logger.error(f"Exception running Co-DETR train: {e_train_run}", exc_info=True)

    if training_successful:
        fold_py_logger.info("Co-DETR native training completed successfully.")
        log_json_file = None # Find .log.json
        for file_in_workdir in sorted(os.listdir(codetr_native_output_dir)):
            if file_in_workdir.endswith(".log.json"):
                log_json_file = os.path.join(codetr_native_output_dir, file_in_workdir); break
        if log_json_file:
            parse_codetr_log_json(log_json_file, os.path.join(custom_eval_output_dir, "epoch_metrics_codetr_comparable.csv"), fold_py_logger)
        else: fold_py_logger.warning(f"No .log.json in {codetr_native_output_dir} for epoch metrics.")

        best_checkpoint_path = None # Find best checkpoint (best_*.pth or latest)
        found_ckpts = [f for f in os.listdir(codetr_native_output_dir) if f.endswith('.pth')]
        best_map_ckpt = [f for f in found_ckpts if "best_bbox_mAP" in f] # MMDetection often saves this
        if best_map_ckpt: best_checkpoint_path = os.path.join(codetr_native_output_dir, best_map_ckpt[0])
        elif found_ckpts: # Fallback to latest modified .pth
            found_ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(codetr_native_output_dir, f)))
            best_checkpoint_path = os.path.join(codetr_native_output_dir, found_ckpts[-1]) if found_ckpts else None


        if best_checkpoint_path:
            fold_py_logger.info(f"Using checkpoint for final eval: {best_checkpoint_path}")
            val_dataset_actual_dir = os.path.join(dataset_root_dir_for_fold, "valid") # Path to the 'valid' subdir
            classes_from_coco_ann = []
            try:
                with open(os.path.join(val_dataset_actual_dir, "_annotations.coco.json"), 'r') as f_coco_val:
                    coco_val_data = json.load(f_coco_val)
                classes_from_coco_ann = [cat['name'] for cat in coco_val_data.get('categories', [])]
                if not classes_from_coco_ann: fold_py_logger.warning("Could not read class names from validation COCO JSON.")
            except Exception as e_read_cls: fold_py_logger.error(f"Failed to read classes from validation COCO JSON: {e_read_cls}")
            
            run_codetr_final_evaluation(fold_config_path, best_checkpoint_path, custom_eval_output_dir, fold_py_logger, assigned_gpu_id_str, val_dataset_actual_dir, classes_from_coco_ann)
        else: fold_py_logger.error(f"No checkpoint in {codetr_native_output_dir} for final eval.")
    else: fold_py_logger.error("Co-DETR Training failed. Skipping metrics parsing and final eval.")

    for handler in list(fold_py_logger.handlers): handler.close(); fold_py_logger.removeHandler(handler)


# --- run_cross_validation_training (Main Orchestrator - Same as before) ---
def run_cross_validation_training():
    logging.info(f"MainProcess: Starting {NUM_FOLDS}-fold CV for {MODEL_DESCRIPTOR} fine-tuning...")
    logging.info(f"MainProcess: Base results directory set to: {BASE_RESULTS_DIR}")
    logging.info(f"MainProcess: COCO Data path template: {BASE_DATA_PATH_TEMPLATE}")
    logging.info(f"MainProcess: Using model config: {BASE_CODETR_CONFIG_FILE}")
    logging.info(f"MainProcess: Custom dataset classes: {NUM_CUSTOM_CLASSES}")
    logging.info(f"MainProcess: Co-DETR Fine-tuning Params (will modify config): {CODETR_TRAIN_PARAMS}")
    logging.warning(f"MainProcess: User MUST ensure CODETR_REPO_PATH and BASE_CODETR_CONFIG_FILE are correct.")
    
    num_available_gpus = 0
    try:
        if torch.cuda.is_available(): num_available_gpus = torch.cuda.device_count()
        logging.info(f"MainProcess: PyTorch detects {num_available_gpus} available CUDA GPU(s).")
        if num_available_gpus == 0 and NUM_FOLDS > 0:
             logging.error("MainProcess: No CUDA GPUs detected! Co-DETR fine-tuning requires GPUs. Each fold will likely fail.")
    except Exception as e: logging.error(f"MainProcess: GPU check error: {e}", exc_info=True); return

    if NUM_FOLDS == 0: logging.info("MainProcess: NUM_FOLDS is 0. No training."); return
    
    gpu_assignment_for_folds = [i % num_available_gpus if num_available_gpus > 0 else None for i in range(NUM_FOLDS)]
    if num_available_gpus == 0 : logging.warning("MainProcess: No GPUs available! Assigning folds to 'None' GPU, Co-DETR will likely fail.")
    elif num_available_gpus < NUM_FOLDS: logging.warning(f"MainProcess: {NUM_FOLDS} folds, but only {num_available_gpus} GPU(s). Folds assigned cyclically.")

    logging.info(f"MainProcess: GPU assignments for folds (0-indexed GPU ID): {gpu_assignment_for_folds}")
    
    training_processes = []
    for i in range(NUM_FOLDS):
        assigned_gpu = gpu_assignment_for_folds[i]
        process_gpu_name_part = str(assigned_gpu) if assigned_gpu is not None else 'CPU_FAIL' # CoDETR likely fails on CPU
        process_name = f"Fold-{i+1}_GPU-{process_gpu_name_part}_{MODEL_DESCRIPTOR}"
        
        logging.info(f"MainProcess: Preparing process {process_name} for Fold {i+1} on assigned GPU index {assigned_gpu}.")
        p = multiprocessing.Process(target=train_single_fold, args=(i, assigned_gpu), name=process_name)
        training_processes.append(p)

    logging.info(f"MainProcess: Starting {len(training_processes)} Co-DETR fine-tuning processes...")
    for p in training_processes: p.start(); logging.info(f"MainProcess: Started {p.name} (PID: {p.pid}).")
    for i, p in enumerate(training_processes):
        p.join()
        logging.info(f"MainProcess: {p.name} for Fold {i+1} completed. Exit code: {p.exitcode}.")
        if p.exitcode != 0: logging.error(f"MainProcess: {p.name} failed. Check logs in {os.path.join(BASE_RESULTS_DIR, f'fold_{i+1}')}")

    logging.info(f"MainProcess: All Co-DETR fine-tuning CV tasks finished.")
    logging.info(f"MainProcess: Check {BASE_RESULTS_DIR} for results.")


if __name__ == '__main__':
    logging.info(f"Starting {MODEL_DESCRIPTOR} CV Fine-tuning script.")
    if not CODETR_REPO_PATH or not os.path.isdir(CODETR_REPO_PATH) or \
       not BASE_CODETR_CONFIG_FILE or not os.path.isfile(BASE_CODETR_CONFIG_FILE):
        logging.error("CRITICAL: CODETR_REPO_PATH or BASE_CODETR_CONFIG_FILE is not valid or not set. Please configure them at the top of the script.")
    elif NUM_CUSTOM_CLASSES <= 0:
        logging.error("CRITICAL: NUM_CUSTOM_CLASSES must be set to a positive integer representing the number of classes in your dataset.")
    else:
        run_cross_validation_training()