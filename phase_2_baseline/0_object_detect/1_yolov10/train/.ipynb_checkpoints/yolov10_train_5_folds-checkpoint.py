import os
import logging
import sys # For stdout/stderr redirection
from ultralytics import YOLO
import multiprocessing
import torch # For GPU check
# import shutil # For potentially cleaning up old results if needed - commented out as not used
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Allow loading of very large images, use with caution.

# --- General Configuration ---
NUM_FOLDS = 5
BASE_DATA_PATH_TEMPLATE = "/blue/hulcr/gmarais/PhD/phase_1_data/2_object_detection_phase_2/ultralytics/cv_iteration_{}"
BASE_RESULTS_DIR = "./results" # Main directory for all cross-validation results
MODEL_NAME_OR_PATH = "yolov10x.pt"

# --- Training Hyperparameters ---
TRAIN_HYPERPARAMS = {
    "epochs": 3,
    "batch": 32,             # Batch size per GPU.
    "workers": 3,            # Number of dataloader workers per GPU
    "patience": 0,
    "close_mosaic": 50,
    "cache": False, # 'disk',         # Ensure you have SUFFICIENT disk space (possibly >1TB)
                             # If not, set to False.
    "imgsz": 640,
    "optimizer": 'AdamW',
    "lr0": 0.0002,
    "lrf": 0.01,
    "weight_decay": 0.0001,
    "dropout": 0.0,
    "pretrained": True,
    "resume": False,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "project": None,         # Will be set per fold
    "name": None,            # Will be set per fold
    "exist_ok": False,
    "verbose": True,         # Let Ultralytics print verbosely
    "save_period": -1,
    "seed": 0,
}

# --- Main Script Logging Setup (Orchestrator Log) ---
main_orchestration_log_dir = os.path.join(BASE_RESULTS_DIR, "orchestration_logs")
os.makedirs(main_orchestration_log_dir, exist_ok=True)

# Clear existing handlers from the root logger for the main process
# This helps avoid duplicate messages if the script is re-run in some environments
root_logger_main = logging.getLogger() # Get the root logger
if root_logger_main.name == 'root': # Only modify the root logger
    if root_logger_main.hasHandlers():
        for handler in list(root_logger_main.handlers): # Iterate over a copy
            root_logger_main.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(main_orchestration_log_dir, "cv_orchestration_main.log")),
        logging.StreamHandler(sys.stdout) # Also print main orchestrator logs to console
    ]
)

# --- Multiprocessing Start Method Configuration ---
try:
    if multiprocessing.get_start_method(allow_none=True) is None or multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    logging.info(f"MainProcess: Multiprocessing start method configured to 'spawn'. Current method: {multiprocessing.get_start_method()}")
except RuntimeError as e:
    logging.warning(f"MainProcess: Could not set multiprocessing start method to 'spawn': {e}. Current method: {multiprocessing.get_start_method(allow_none=True)}")


def train_single_fold(fold_idx, assigned_gpu_id):
    """
    Trains the YOLO model for a specific fold.
    """
    fold_number_display = fold_idx + 1
    current_process_name = multiprocessing.current_process().name # e.g., Fold-1_GPU-0

    fold_results_project_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{fold_number_display}")
    os.makedirs(fold_results_project_dir, exist_ok=True)

    # --- Per-Fold Python Logger Setup ---
    fold_py_logger = logging.getLogger(current_process_name) # Unique logger name
    fold_py_logger.propagate = False # Do not pass to the main process's root logger
    fold_py_logger.setLevel(logging.INFO)

    if fold_py_logger.hasHandlers(): # Clear any pre-existing handlers for this logger instance
        for handler in list(fold_py_logger.handlers):
            fold_py_logger.removeHandler(handler)
            handler.close()


    py_log_file = os.path.join(fold_results_project_dir, f"script_log_fold_{fold_number_display}_gpu_{assigned_gpu_id}.log")
    file_handler = logging.FileHandler(py_log_file, mode='a')
    formatter = logging.Formatter(f'%(asctime)s - {current_process_name} - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    fold_py_logger.addHandler(file_handler)
    # Optional: If you want this fold's specific script logs to also go to its stdout (which will be redirected)
    # stream_handler_fold = logging.StreamHandler(sys.stdout)
    # stream_handler_fold.setFormatter(formatter)
    # fold_py_logger.addHandler(stream_handler_fold)

    fold_py_logger.info(f"Initializing training for Fold {fold_number_display} on GPU {assigned_gpu_id}.")

    data_yaml_path = os.path.join(BASE_DATA_PATH_TEMPLATE.format(fold_number_display), "data.yaml")

    run_name_parts = [
        f"yolo_{MODEL_NAME_OR_PATH.replace('.pt','')}",
        f"e{TRAIN_HYPERPARAMS.get('epochs', 'N')}",
        f"b{TRAIN_HYPERPARAMS.get('batch', 'N')}", # CRITICAL: batch=63 is very high!
        f"imgsz{TRAIN_HYPERPARAMS.get('imgsz', 'N')}"
    ]
    fold_run_name = "_".join(run_name_parts)

    fold_py_logger.info(f"Data YAML: {data_yaml_path}")
    fold_py_logger.info(f"Ultralytics project dir (where Ultralytics saves its run): {fold_results_project_dir}")
    fold_py_logger.info(f"Ultralytics run name: {fold_run_name}")

    if not os.path.exists(data_yaml_path):
        fold_py_logger.error(f"CRITICAL - Data YAML file not found: {data_yaml_path}. Skipping this fold.")
        return

    # --- Redirect stdout/stderr for this fold's process ---
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    # This will capture Ultralytics' direct print outputs
    stdout_stderr_log_file = os.path.join(fold_results_project_dir, f"ultralytics_output_fold_{fold_number_display}_gpu_{assigned_gpu_id}.log")

    try:
        with open(stdout_stderr_log_file, 'w') as f_log:
            sys.stdout = f_log
            sys.stderr = f_log

            fold_py_logger.info("Python-level stdout and stderr for this fold are now redirected to: " + stdout_stderr_log_file)
            print(f"--- This is the start of redirected stdout/stderr for {current_process_name} ---") # For the redirected file

            model = YOLO(MODEL_NAME_OR_PATH)
            fold_py_logger.info(f"Loaded base model {MODEL_NAME_OR_PATH}.") # Goes to script_log_...
            print(f"Ultralytics output: Loaded base model {MODEL_NAME_OR_PATH}.") # Goes to ultralytics_output_...

            device_to_use = str(assigned_gpu_id) if assigned_gpu_id is not None else 'cpu'

            current_train_params = TRAIN_HYPERPARAMS.copy()
            current_train_params['data'] = data_yaml_path
            current_train_params['device'] = device_to_use
            current_train_params['project'] = fold_results_project_dir # Ultralytics will save under this
            current_train_params['name'] = fold_run_name      # This will be the specific experiment subfolder

            # Log parameters that will be used by model.train()
            param_log_str_display = {k: v for k, v in current_train_params.items()}
            fold_py_logger.info(f"Starting Ultralytics training with effective hyperparameters: {param_log_str_display}")
            print(f"Ultralytics output: Starting training with effective hyperparameters: {param_log_str_display}")

            results = model.train(**current_train_params)

            fold_py_logger.info("Training completed successfully.") # Goes to script_log_...
            fold_py_logger.info(f"All Ultralytics results, metrics, and model weights saved in: {results.save_dir}")
            print(f"Ultralytics output: Training completed. Results in: {results.save_dir}") # Goes to ultralytics_output_...

    except Exception as e:
        # Log exception to this fold's specific Python logger
        fold_py_logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Also print to original stderr so it's visible in Slurm's main error log if redirection fails
        sys.stdout = orig_stdout # Restore before printing critical error to main streams
        sys.stderr = orig_stderr
        print(f"CRITICAL ERROR in {current_process_name}: {e}\nSee {py_log_file} and {stdout_stderr_log_file} for details.", file=sys.stderr)
        # Re-raise the exception so the multiprocessing parent knows the process failed
        raise
    finally:
        # Ensure stdout and stderr are restored
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        fold_py_logger.info("Restored stdout/stderr for this fold process.")
        # Close per-fold Python logger handlers
        for handler in list(fold_py_logger.handlers):
            handler.close()
            fold_py_logger.removeHandler(handler)


def run_cross_validation_training():
    logging.info(f"MainProcess: Starting {NUM_FOLDS}-fold cross-validation training orchestrator...")
    logging.info(f"MainProcess: Base results directory set to: {BASE_RESULTS_DIR}")
    logging.info(f"MainProcess: Data path template: {BASE_DATA_PATH_TEMPLATE}")
    logging.info(f"MainProcess: Using model: {MODEL_NAME_OR_PATH}")
    logging.info(f"MainProcess: Global Training Hyperparameters: {TRAIN_HYPERPARAMS}")

    num_available_gpus = 0
    try:
        if torch.cuda.is_available():
            num_available_gpus = torch.cuda.device_count()
            logging.info(f"MainProcess: PyTorch detects {num_available_gpus} available CUDA GPU(s).")
        else:
            logging.warning("MainProcess: PyTorch reports CUDA not available. Training will attempt CPU if device is not set.")

        if num_available_gpus == 0 and NUM_FOLDS > 0 :
            logging.error("MainProcess: No CUDA GPUs detected by PyTorch. Cannot proceed with GPU-based training for the requested folds if GPUs are required.")
            # Allow to proceed if user intends CPU, device will be 'cpu'
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
                f"Folds will be assigned to available GPUs cyclically."
            )
        gpu_assignment_for_folds = [i % num_available_gpus for i in range(NUM_FOLDS)]

    logging.info(f"MainProcess: GPU assignments for folds (Fold Index -> Assigned GPU ID or None for CPU):")
    for i in range(NUM_FOLDS):
        assigned_gpu_for_log = gpu_assignment_for_folds[i] if gpu_assignment_for_folds[i] is not None else 'CPU'
        logging.info(f"MainProcess:   Fold {i+1} (0-indexed: {i}) --> GPU {assigned_gpu_for_log}")

    training_processes = []
    for i in range(NUM_FOLDS):
        fold_idx_0_based = i
        assigned_gpu = gpu_assignment_for_folds[i]
        process_gpu_name_part = str(assigned_gpu) if assigned_gpu is not None else 'CPU'
        process_name = f"Fold-{fold_idx_0_based+1}_GPU-{process_gpu_name_part}"

        logging.info(f"MainProcess: Preparing to launch training process {process_name} for Fold {fold_idx_0_based+1} on GPU {process_gpu_name_part if assigned_gpu is not None else 'CPU'}.")
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
        p.join() # Wait for the process to complete
        logging.info(f"MainProcess: Process {p.name} (PID: {p.pid}) for Fold {i+1} has completed. Exit code: {p.exitcode}.")
        if p.exitcode != 0:
            logging.error(f"MainProcess: Process {p.name} for Fold {i+1} (PID: {p.pid}) exited with a non-zero code ({p.exitcode}). "
                          f"Check its dedicated log files in {os.path.join(BASE_RESULTS_DIR, f'fold_{i+1}')} and the Slurm error file.")

    logging.info(f"MainProcess: All {NUM_FOLDS}-fold cross-validation training tasks have finished processing.")
    logging.info(f"MainProcess: Please check individual fold directories under {BASE_RESULTS_DIR} for detailed logs and training results.")

if __name__ == '__main__':
    # Important pre-run checks are listed in comments in the original script.
    # Ensure data paths, model, hyperparameters, and system resources are adequate.
    run_cross_validation_training()