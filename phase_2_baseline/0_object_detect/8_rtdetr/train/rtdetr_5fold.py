import os
import logging
import sys  # For stdout/stderr redirection
from ultralytics import RTDETR # Using specific RTDETR class
import multiprocessing
import torch  # For GPU check
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Allow loading of very large images

# --- General Configuration ---
NUM_FOLDS = 5
BASE_DATA_PATH_TEMPLATE = "/blue/hulcr/gmarais/PhD/phase_1_data/2_object_detection_phase_2/ultralytics/cv_iteration_{}"
BASE_RESULTS_DIR = "./results_rtdetr-x"  # Main directory for all cross-validation results
MODEL_NAME_OR_PATH = "rtdetr-x.pt"  # RTDETR-x weights

# --- Training Hyperparameters ---
# Note: "end2end": True is crucial for RT-DETR to use appropriate loss and avoid 'stride' errors.
TRAIN_HYPERPARAMS = {
    "epochs": 3,
    "batch": 16,            # Batch size per GPU. RTDETR can be memory intensive.
    "workers": 3,           # Number of dataloader workers per GPU
    "patience": 0,          # Early stopping patience (0 to disable)
    "close_mosaic": 50,     # Epochs to disable mosaic augmentation towards the end
    "cache": False,         # Whether to cache images (RAM or disk). Set to 'disk' or True if RAM is insufficient.
    "imgsz": 640,           # Image size
    "optimizer": 'AdamW',   # Optimizer
    "lr0": 1e-4,            # Initial learning rate
    "lrf": 0.01,            # Final learning rate factor (lr0 * lrf)
    "weight_decay": 1e-4,   # Optimizer weight decay
    "dropout": 0.0,         # Dropout rate (if applicable to the model architecture)
    "pretrained": True,     # Start from pretrained weights specified in MODEL_NAME_OR_PATH
    "resume": False,        # Resume from a specific checkpoint (path if True, else bool)

    # Augmentation parameters
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
    "mosaic": 1.0,          # Mosaic augmentation (typically 1.0 for YOLO, RT-DETR might differ)
    "mixup": 0.0,           # Mixup augmentation
    "copy_paste": 0.0,      # Copy-paste augmentation

    # "end2end": True,        # Crucial for RT-DETR: uses DETR-style loss, avoids 'stride' issues.

    # Checkpoint & metrics saving
    "save": True,           # Save training checkpoints and final model
    "save_period": 1,       # Save checkpoint every N epochs (-1 for last only)
    "exist_ok": True,       # Overwrite existing Ultralytics project/name directory
    "verbose": True,        # Print verbose output during training
    "seed": 0,              # Global random seed

    # Placeholders, will be set per fold:
    "project": None,        # Ultralytics project directory (base for runs)
    "name": None,           # Ultralytics run name (subdirectory within project)
}

# --- Main Script Logging Setup (Orchestrator Log) ---
main_orchestration_log_dir = os.path.join(BASE_RESULTS_DIR, "orchestration_logs")
os.makedirs(main_orchestration_log_dir, exist_ok=True)

# Clear existing handlers from the root logger for the main process
root_logger_main = logging.getLogger()  # Get the root logger
if root_logger_main.name == 'root':  # Only modify the root logger
    if root_logger_main.hasHandlers():
        for handler in list(root_logger_main.handlers):  # Iterate over a copy
            root_logger_main.removeHandler(handler)
            handler.close() # Ensure handlers are closed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(main_orchestration_log_dir, f"cv_orchestration_main_{MODEL_NAME_OR_PATH.replace('.pt','')}.log")),
        logging.StreamHandler(sys.stdout)  # Also print main orchestrator logs to console
    ]
)

# --- Multiprocessing Start Method Configuration ---
try:
    if multiprocessing.get_start_method(allow_none=True) is None or multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    logging.info(f"MainProcess: Multiprocessing start method configured to 'spawn'. Current method: {multiprocessing.get_start_method()}")
except RuntimeError as e:
    # This can happen if the start method has already been set and 'force=True' is not honored or if called at the wrong time.
    logging.warning(f"MainProcess: Could not set multiprocessing start method to 'spawn': {e}. Current method: {multiprocessing.get_start_method(allow_none=True)}")


def train_single_fold(fold_idx, assigned_gpu_id):
    """
    Trains the RTDETR model for a specific fold.
    """
    fold_number_display = fold_idx + 1
    current_process_name = multiprocessing.current_process().name  # e.g., Fold-1_GPU-0

    # Directory for this fold's results (Ultralytics project dir for this fold)
    fold_results_project_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{fold_number_display}")
    os.makedirs(fold_results_project_dir, exist_ok=True)

    # --- Per-Fold Python Logger Setup ---
    # This logger is for this script's specific logging within the fold, not Ultralytics' output
    fold_py_logger = logging.getLogger(current_process_name)  # Unique logger name for this process
    fold_py_logger.propagate = False  # Do not pass to the main process's root logger
    fold_py_logger.setLevel(logging.INFO)

    if fold_py_logger.hasHandlers():  # Clear any pre-existing handlers for this logger instance
        for handler in list(fold_py_logger.handlers):
            fold_py_logger.removeHandler(handler)
            handler.close()

    py_log_file = os.path.join(fold_results_project_dir, f"script_log_fold_{fold_number_display}_gpu_{assigned_gpu_id}.log")
    file_handler = logging.FileHandler(py_log_file, mode='w') # 'w' to overwrite for a fresh run of this fold
    formatter = logging.Formatter(f'%(asctime)s - {current_process_name} - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    fold_py_logger.addHandler(file_handler)

    fold_py_logger.info(f"Initializing training for Fold {fold_number_display} on GPU {assigned_gpu_id if assigned_gpu_id is not None else 'CPU'} with model {MODEL_NAME_OR_PATH}.")

    data_yaml_path = os.path.join(BASE_DATA_PATH_TEMPLATE.format(fold_number_display), "data.yaml")

    # Define the Ultralytics run name for this specific fold's training
    run_name_parts = [
        f"{MODEL_NAME_OR_PATH.replace('.pt','')}",
        f"e{TRAIN_HYPERPARAMS.get('epochs', 'N')}",
        f"b{TRAIN_HYPERPARAMS.get('batch', 'N')}",
        f"imgsz{TRAIN_HYPERPARAMS.get('imgsz', 'N')}"
    ]
    fold_run_name = "_".join(run_name_parts)

    fold_py_logger.info(f"Data YAML: {data_yaml_path}")
    fold_py_logger.info(f"Ultralytics project dir (base for this fold's runs): {fold_results_project_dir}")
    fold_py_logger.info(f"Ultralytics run name (specific training run): {fold_run_name}")

    if not os.path.exists(data_yaml_path):
        fold_py_logger.error(f"CRITICAL - Data YAML file not found: {data_yaml_path}. Skipping this fold.")
        # Close handler for this fold's logger
        for handler_to_close in list(fold_py_logger.handlers):
            handler_to_close.close()
            fold_py_logger.removeHandler(handler_to_close)
        return

    # --- Redirect stdout/stderr for this fold's Ultralytics process ---
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    # This log captures everything Ultralytics prints to stdout/stderr
    ultralytics_output_log_file = os.path.join(fold_results_project_dir, f"ultralytics_output_fold_{fold_number_display}_gpu_{assigned_gpu_id}.log")

    try:
        with open(ultralytics_output_log_file, 'w') as f_log: # 'w' to overwrite for a fresh run
            sys.stdout = f_log
            sys.stderr = f_log

            fold_py_logger.info("Python-level stdout and stderr for this fold (for Ultralytics) are now redirected to: " + ultralytics_output_log_file)
            print(f"--- This is the start of redirected Ultralytics stdout/stderr for {current_process_name} ---")

            model = RTDETR(MODEL_NAME_OR_PATH) # Load the specific RTDETR model
            fold_py_logger.info(f"Loaded base model {MODEL_NAME_OR_PATH} using ultralytics.RTDETR class.")
            print(f"Ultralytics output: Loaded base model {MODEL_NAME_OR_PATH} using RTDETR class.")

            device_to_use = str(assigned_gpu_id) if assigned_gpu_id is not None else 'cpu'

            current_train_params = TRAIN_HYPERPARAMS.copy()
            current_train_params['data'] = data_yaml_path
            current_train_params['device'] = device_to_use
            current_train_params['project'] = fold_results_project_dir # Ultralytics project for this fold
            current_train_params['name'] = fold_run_name      # Ultralytics run name

            param_log_str_display = {k: v for k, v in current_train_params.items()}
            fold_py_logger.info(f"Starting Ultralytics training with effective hyperparameters: {param_log_str_display}")
            print(f"Ultralytics output: Starting training with effective hyperparameters: {param_log_str_display}")

            results = model.train(**current_train_params)

            fold_py_logger.info("Training completed successfully.")
            fold_py_logger.info(f"All Ultralytics results, metrics, and model weights saved in: {results.save_dir}") # results.save_dir is specific path like fold_X/run_name
            print(f"Ultralytics output: Training completed. Results in: {results.save_dir}")

    except Exception as e:
        # Log to this fold's specific Python logger first
        fold_py_logger.error(f"An error occurred during training for {current_process_name}: {e}", exc_info=True)
        # Restore stdout/stderr before printing to console, so it appears on main console/log
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        # This message will go to the main process's stdout/stderr (console or main orchestrator log)
        print(f"CRITICAL ERROR in {current_process_name}: {e}\nSee script log {py_log_file} and Ultralytics log {ultralytics_output_log_file} for details.", file=sys.stderr)
        raise # Re-raise the exception to be caught by the main process if necessary
    finally:
        # Always restore stdout/stderr
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        fold_py_logger.info("Restored stdout/stderr for this fold process.")
        # Close and remove handlers for this fold's specific logger
        for handler_to_close in list(fold_py_logger.handlers):
            handler_to_close.close()
            fold_py_logger.removeHandler(handler_to_close)


def run_cross_validation_training():
    logging.info(f"MainProcess: Starting {NUM_FOLDS}-fold cross-validation training orchestrator for {MODEL_NAME_OR_PATH}...")
    logging.info(f"MainProcess: Base results directory set to: {BASE_RESULTS_DIR}")
    logging.info(f"MainProcess: Data path template: {BASE_DATA_PATH_TEMPLATE}")
    logging.info(f"MainProcess: Using model: {MODEL_NAME_OR_PATH}")
        logging.info(f"MainProcess: Global Training Hyperparameters (see TRAIN_HYPERPARAMS in script for full list): epochs={TRAIN_HYPERPARAMS['epochs']}, batch={TRAIN_HYPERPARAMS['batch']}, imgsz={TRAIN_HYPERPARAMS['imgsz']}")

    # logging.info(f"MainProcess: Global Training Hyperparameters (see TRAIN_HYPERPARAMS in script for full list): epochs={TRAIN_HYPERPARAMS['epochs']}, batch={TRAIN_HYPERPARAMS['batch']}, imgsz={TRAIN_HYPERPARAMS['imgsz']}, end2end={TRAIN_HYPERPARAMS['end2end']}")
    # if not TRAIN_HYPERPARAMS['end2end']:
    #     logging.warning("MainProcess: 'end2end' is False. For RT-DETR, 'end2end=True' is typically required to use the correct loss function and avoid 'stride' related errors.")


    num_available_gpus = 0
    try:
        if torch.cuda.is_available():
            num_available_gpus = torch.cuda.device_count()
            logging.info(f"MainProcess: PyTorch detects {num_available_gpus} available CUDA GPU(s).")
            for i in range(num_available_gpus):
                logging.info(f"MainProcess: GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logging.warning("MainProcess: PyTorch reports CUDA not available. Training will attempt CPU if device is not explicitly set to a GPU.")

        if num_available_gpus == 0 and NUM_FOLDS > 0:
             logging.warning("MainProcess: No CUDA GPUs detected by PyTorch. All folds will be assigned to CPU.")
    except Exception as e:
        logging.error(f"MainProcess: Could not verify GPU count using PyTorch: {e}. Ensure PyTorch is installed correctly with CUDA support if GPUs are intended.")
        # Decide if to proceed with CPU or exit. For now, will let it proceed with num_available_gpus = 0.
        # return # Uncomment to exit if GPU check fails catastrophically

    if NUM_FOLDS == 0:
        logging.info("MainProcess: NUM_FOLDS is 0. No training will be performed.")
        return

    if num_available_gpus == 0:
        logging.warning("MainProcess: No GPUs available. Assigning all folds to run on CPU (device will be 'cpu').")
        gpu_assignment_for_folds = [None] * NUM_FOLDS # None signifies CPU for Ultralytics device setting
    else:
        if num_available_gpus < NUM_FOLDS:
            logging.warning(
                f"MainProcess: Requested {NUM_FOLDS} folds, but only {num_available_gpus} GPU(s) are detected. "
                f"Folds will be assigned to available GPUs cyclically."
            )
        gpu_assignment_for_folds = [i % num_available_gpus for i in range(NUM_FOLDS)]

    logging.info(f"MainProcess: GPU assignments for folds (Fold Index -> Assigned GPU ID or 'CPU'):")
    for i in range(NUM_FOLDS):
        assigned_gpu_for_log = gpu_assignment_for_folds[i] if gpu_assignment_for_folds[i] is not None else 'CPU'
        logging.info(f"MainProcess:   Fold {i+1} (0-indexed: {i}) --> GPU {assigned_gpu_for_log}")

    training_processes = []
    for i in range(NUM_FOLDS):
        fold_idx_0_based = i
        assigned_gpu = gpu_assignment_for_folds[i]
        process_gpu_name_part = str(assigned_gpu) if assigned_gpu is not None else 'CPU'
        process_name = f"Fold-{fold_idx_0_based+1}_GPU-{process_gpu_name_part}"

        logging.info(f"MainProcess: Preparing to launch training process {process_name} for Fold {fold_idx_0_based+1} on GPU/Device '{process_gpu_name_part}'.")
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
            logging.error(f"MainProcess: Process {p.name} for Fold {i+1} (PID: {p.pid}) exited with a non-zero code ({p.exitcode}). "
                          f"Check its dedicated log files in {os.path.join(BASE_RESULTS_DIR, f'fold_{i+1}')} and the main error output if any.")

    logging.info(f"MainProcess: All {NUM_FOLDS}-fold cross-validation training tasks have finished processing for {MODEL_NAME_OR_PATH}.")
    logging.info(f"MainProcess: Please check individual fold directories under {BASE_RESULTS_DIR} for detailed logs and training results.")

if __name__ == '__main__':
    logging.info(f"MainProcess: Starting RT-DETR training script. Ultralytics version: {torch.hub.ultralytics.VERSION if hasattr(torch.hub, 'ultralytics') else 'unknown (torch.hub.ultralytics not found)'}, PyTorch version: {torch.__version__}")
    logging.info(f"MainProcess: Ensure '{MODEL_NAME_OR_PATH}' is accessible and you are in the correct Conda environment ('EEL' based on original paths).")
    run_cross_validation_training()