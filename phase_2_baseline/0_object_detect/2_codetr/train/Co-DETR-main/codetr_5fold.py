import os
import logging
import sys # For stdout/stderr redirection
import multiprocessing
import torch # For GPU check
import subprocess # For running MMDetection's train script
# from mmcv import Config # Only needed if you want to load/modify config object directly in this script
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Allow loading of very large images, use with caution.

# --- General Configuration ---
NUM_FOLDS = 5
# IMPORTANT: Update this path to your COCO-formatted data for each fold
BASE_DATA_PATH_TEMPLATE = "/blue/hulcr/gmarais/PhD/IBBI_work/phase_1_data/2_object_detection_phase_2/coco/cv_iteration_{}"
BASE_RESULTS_DIR = "./results_codetr" # Main directory for all cross-validation results

# IMPORTANT: Choose an appropriate Co-DETR base configuration file
# This should be a path relative to the root of the Co-DETR repository
CODETR_BASE_CONFIG_PATH = "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"
# If you have a specific Co-DETR checkpoint to start from (not just backbone pretraining), set its path here or ensure 'load_from' in the base config is correct.
# CODETR_PRETRAINED_WEIGHTS_PATH = None # Example: "path/to/your/co_detr_checkpoint.pth"

# --- Training Hyperparameters (adjust as needed for Co-DETR) ---
# Note: Many Co-DETR hyperparameters are set within its config file.
# These are overrides or global settings for the training process.
TRAIN_HYPERPARAMS = {
    "epochs": 3,             # Mapped to runner.max_epochs
    "batch": 2,              # Mapped to data.samples_per_gpu (adjust based on GPU memory)
    "workers": 2,            # Mapped to data.workers_per_gpu
    "patience": 0,           # MMDetection uses different early stopping, not directly mapped
    "close_mosaic": 50,      # Mosaic is configured in MMDetection's data pipeline, complex to toggle this way
    "cache": False,          # Mapped to data.train.dataset.cache_images or similar (depends on config)
    "imgsz": 640,            # Mapped to train_pipeline Resize transform's img_scale
    "optimizer": 'AdamW',    # Usually set in base config's optimizer.type
    "lr0": 0.0001,           # Mapped to optimizer.lr
    "lrf": 0.01,             # Learning rate factor for final LR, MMDetection's lr_config is more complex. This is not directly mapped.
    "weight_decay": 0.0005,  # Mapped to optimizer.weight_decay
    "dropout": 0.0,          # Usually part of model architecture in config, not a simple override.
    "pretrained": True,      # True will try to use `load_from` in config or CODETR_PRETRAINED_WEIGHTS_PATH if set
    "resume": False,         # Set to True and provide path in CODETR_RESUME_FROM_PATH if needed
    # "CODETR_RESUME_FROM_PATH": None, # Path to a checkpoint for resuming
    # Augmentations: Best configured in the base Co-DETR config file's pipeline.
    # "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1,
    # "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
    # "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
    "project": None,         # Used to construct work_dir
    "name": None,            # Used to construct work_dir (experiment name part)
    "exist_ok": False,       # For os.makedirs, MMDetection's work_dir handles its own creation.
    "verbose": True,         # MMDetection's train.py is generally verbose.
    "save_period": 1,        # Mapped to checkpoint_config.interval (save checkpoint every N epochs)
    "seed": 0,               # Mapped to --seed for train.py
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
        logging.FileHandler(os.path.join(main_orchestration_log_dir, "cv_orchestration_main_codetr.log")),
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


def train_single_fold(fold_idx, assigned_gpu_id):
    fold_number_display = fold_idx + 1
    current_process_name = multiprocessing.current_process().name

    fold_results_project_dir = os.path.join(BASE_RESULTS_DIR, f"fold_{fold_number_display}")
    os.makedirs(fold_results_project_dir, exist_ok=True)

    fold_py_logger = logging.getLogger(current_process_name)
    fold_py_logger.propagate = False
    fold_py_logger.setLevel(logging.INFO)

    if fold_py_logger.hasHandlers():
        for handler in list(fold_py_logger.handlers):
            fold_py_logger.removeHandler(handler)
            handler.close()

    py_log_file = os.path.join(fold_results_project_dir, f"script_log_fold_{fold_number_display}_gpu_{assigned_gpu_id if assigned_gpu_id is not None else 'cpu'}.log")
    file_handler = logging.FileHandler(py_log_file, mode='a')
    formatter = logging.Formatter(f'%(asctime)s - {current_process_name} - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    fold_py_logger.addHandler(file_handler)

    fold_py_logger.info(f"Initializing training for Fold {fold_number_display} on GPU {assigned_gpu_id if assigned_gpu_id is not None else 'CPU'}.")

    # Data paths for COCO format (adjust if your annotation names differ)
    # These are relative to the data_root which will be set by BASE_DATA_PATH_TEMPLATE
    train_ann_file = 'annotations/instances_train.json'
    train_img_prefix = 'train/'
    val_ann_file = 'annotations/instances_val.json' # Assuming validation set is used for evaluation
    val_img_prefix = 'val/'

    # Check if the base data directory for the fold exists
    fold_data_path = BASE_DATA_PATH_TEMPLATE.format(fold_number_display)
    if not os.path.exists(fold_data_path):
        fold_py_logger.error(f"CRITICAL - Base data directory not found: {fold_data_path}. Skipping this fold.")
        return
    # Further checks for annotation files can be added if needed, but MMDetection will error out if not found.

    # --- Experiment Naming for MMDetection (used in work_dir path) ---
    model_config_name = CODETR_BASE_CONFIG_PATH.split('/')[-1].replace('.py', '')
    run_name_parts = [
        model_config_name,
        f"e{TRAIN_HYPERPARAMS.get('epochs', 'N')}",
        f"b{TRAIN_HYPERPARAMS.get('batch', 'N')}",
        f"imgsz{TRAIN_HYPERPARAMS.get('imgsz', 'N')}"
    ]
    fold_run_name = "_".join(run_name_parts) # This will be the experiment name within the fold's project dir

    work_dir_for_fold = os.path.join(fold_results_project_dir, fold_run_name)
    os.makedirs(work_dir_for_fold, exist_ok=True) # MMDetection also creates it, but good practice

    fold_py_logger.info(f"MMDetection base config: {CODETR_BASE_CONFIG_PATH}")
    fold_py_logger.info(f"MMDetection work_dir (results, checkpoints, logs): {work_dir_for_fold}")

    # --- Redirect stdout/stderr for this fold's process (captures MMDetection output) ---
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    stdout_stderr_log_file = os.path.join(fold_results_project_dir, f"mmdetection_output_fold_{fold_number_display}_gpu_{assigned_gpu_id if assigned_gpu_id is not None else 'cpu'}.log")

    try:
        with open(stdout_stderr_log_file, 'w') as f_log:
            sys.stdout = f_log
            sys.stderr = f_log
            fold_py_logger.info("Python-level stdout and stderr for this fold are now redirected to: " + stdout_stderr_log_file)
            print(f"--- This is the start of redirected stdout/stderr for {current_process_name} ---")

            # --- Prepare cfg-options for MMDetection ---
            cfg_options_dict = {}
            cfg_options_dict['data.train.data_root'] = fold_data_path
            cfg_options_dict['data.train.ann_file'] = train_ann_file
            cfg_options_dict['data.train.img_prefix'] = train_img_prefix

            cfg_options_dict['data.val.data_root'] = fold_data_path
            cfg_options_dict['data.val.ann_file'] = val_ann_file
            cfg_options_dict['data.val.img_prefix'] = val_img_prefix
            
            # MMDetection uses data.test for evaluation during training if `evaluation.test_mode=False` (default usually)
            # or if `workflow = [('train', N), ('val', 1)]` includes val.
            # Here, we set data.test to be the same as data.val for simplicity.
            cfg_options_dict['data.test.data_root'] = fold_data_path
            cfg_options_dict['data.test.ann_file'] = val_ann_file
            cfg_options_dict['data.test.img_prefix'] = val_img_prefix


            cfg_options_dict['runner.max_epochs'] = TRAIN_HYPERPARAMS['epochs']
            cfg_options_dict['data.samples_per_gpu'] = TRAIN_HYPERPARAMS['batch']
            cfg_options_dict['data.workers_per_gpu'] = TRAIN_HYPERPARAMS['workers']
            
            if 'lr0' in TRAIN_HYPERPARAMS:
                 cfg_options_dict['optimizer.lr'] = TRAIN_HYPERPARAMS['lr0']
            if 'weight_decay' in TRAIN_HYPERPARAMS:
                cfg_options_dict['optimizer.weight_decay'] = TRAIN_HYPERPARAMS['weight_decay']

            # Image size (imgsz) - this depends on the pipeline structure in the base config.
            # Assuming the first Resize op in train_pipeline and test_pipeline controls this.
            # Example: cfg_options_dict['train_pipeline.0.img_scale'] = (TRAIN_HYPERPARAMS['imgsz'], TRAIN_HYPERPARAMS['imgsz'])
            # This is fragile. It's better if the base config is already set for the desired imgsz or if you create a derived config.
            # For now, this part is commented out, assuming base config handles it or it's manually adjusted in base config.
            # if 'imgsz' in TRAIN_HYPERPARAMS:
            #     img_size = (TRAIN_HYPERPARAMS['imgsz'], TRAIN_HYPERPARAMS['imgsz'])
            #     # This assumes 'Resize' is at specific indices and has 'img_scale'
            #     # You might need to inspect your CODETR_BASE_CONFIG_PATH to find the correct paths
            #     cfg_options_dict['train_pipeline.1.img_scale'] = img_size # Example index
            #     cfg_options_dict['test_pipeline.0.img_scale'] = img_scale  # Example index


            if TRAIN_HYPERPARAMS.get('save_period', -1) > 0:
                cfg_options_dict['checkpoint_config.interval'] = TRAIN_HYPERPARAMS['save_period']

            # `load_from` for pretrained model weights
            # if TRAIN_HYPERPARAMS.get('pretrained') and CODETR_PRETRAINED_WEIGHTS_PATH:
            #     cfg_options_dict['load_from'] = CODETR_PRETRAINED_WEIGHTS_PATH
            # elif TRAIN_HYPERPARAMS.get('pretrained'):
                # Rely on 'load_from' in the CODETR_BASE_CONFIG_PATH or MMDetection's auto-download for backbones
                # No explicit cfg_option needed if base config handles it.
            #    pass
            
            # `resume_from` for resuming training
            # if TRAIN_HYPERPARAMS.get('resume') and TRAIN_HYPERPARAMS.get('CODETR_RESUME_FROM_PATH'):
            #    cfg_options_dict['resume_from'] = TRAIN_HYPERPARAMS['CODETR_RESUME_FROM_PATH']


            # --- Construct MMDetection Training Command ---
            cmd = [
                sys.executable, # Ensures using the correct python interpreter
                "tools/train.py",
                CODETR_BASE_CONFIG_PATH,
                "--work-dir", work_dir_for_fold,
            ]

            # GPU assignment
            env = os.environ.copy()
            if assigned_gpu_id is not None:
                # MMDetection's train.py typically uses CUDA_VISIBLE_DEVICES
                # If --gpu-ids is used, it further restricts from those visible.
                # For simplicity with single-GPU-per-process model:
                env['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu_id)
                cmd.extend(["--gpus", "1"]) # Tell MMDetection to use 1 GPU from the visible ones
                # Alternative for some MMDetection versions/setups: cmd.extend(["--gpu-ids", str(assigned_gpu_id)])
            else: # CPU training
                env['CUDA_VISIBLE_DEVICES'] = "-1" # Explicitly ask for CPU
                # No --gpus or --gpu-ids needed for CPU

            # Add --cfg-options
            if cfg_options_dict:
                cmd.append("--cfg-options")
                option_strings = [f"{k}={v}" for k, v in cfg_options_dict.items()]
                cmd.extend(option_strings)
            
            # Add seed for reproducibility if specified
            if TRAIN_HYPERPARAMS.get('seed') is not None:
                 cmd.extend(['--seed', str(TRAIN_HYPERPARAMS['seed']), '--deterministic'])


            param_log_str_display = {**cfg_options_dict, "base_config": CODETR_BASE_CONFIG_PATH, "gpu_for_cmd": assigned_gpu_id}
            fold_py_logger.info(f"Effective MMDetection parameters and command structure: {param_log_str_display}")
            fold_py_logger.info(f"Executing Co-DETR training command: {' '.join(cmd)}")
            print(f"MMDetection output: Executing command: {' '.join(cmd)}") # For the redirected file
            print(f"MMDetection output: Effective cfg_options: {cfg_options_dict}")


            # Execute the training command
            process_result = subprocess.run(cmd, check=False, env=env) # check=False to handle exit code manually

            if process_result.returncode == 0:
                fold_py_logger.info("MMDetection training process completed successfully.")
                fold_py_logger.info(f"All Co-DETR results, metrics, and model weights saved in: {work_dir_for_fold}")
                print(f"MMDetection output: Training completed. Results in: {work_dir_for_fold}")
            else:
                fold_py_logger.error(f"MMDetection training process failed with exit code {process_result.returncode}.")
                # The actual error message would have been printed to the stdout/stderr log file.
                raise subprocess.CalledProcessError(process_result.returncode, cmd)


    except Exception as e:
        sys.stdout = orig_stdout # Restore before printing critical error to main streams
        sys.stderr = orig_stderr
        fold_py_logger.error(f"An error occurred during Co-DETR training for fold {fold_number_display}: {e}", exc_info=True)
        print(f"CRITICAL ERROR in {current_process_name}: {e}\nSee {py_log_file} and {stdout_stderr_log_file} for details.", file=sys.stderr)
        raise
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        fold_py_logger.info("Restored stdout/stderr for this fold process.")
        for handler in list(fold_py_logger.handlers):
            handler.close()
            fold_py_logger.removeHandler(handler)


def run_cross_validation_training():
    logging.info(f"MainProcess: Starting {NUM_FOLDS}-fold cross-validation training for Co-DETR...")
    logging.info(f"MainProcess: Base results directory set to: {BASE_RESULTS_DIR}")
    logging.info(f"MainProcess: Data path template: {BASE_DATA_PATH_TEMPLATE}")
    logging.info(f"MainProcess: Using Co-DETR base config: {CODETR_BASE_CONFIG_PATH}")
    logging.info(f"MainProcess: Global Training Hyperparameter Overrides: {TRAIN_HYPERPARAMS}")

    num_available_gpus = 0
    try:
        if torch.cuda.is_available():
            num_available_gpus = torch.cuda.device_count()
            logging.info(f"MainProcess: PyTorch detects {num_available_gpus} available CUDA GPU(s).")
        else:
            logging.warning("MainProcess: PyTorch reports CUDA not available. Training will use CPU if no GPUs assigned.")
    except Exception as e:
        logging.error(f"MainProcess: Could not verify GPU count using PyTorch: {e}. Assuming 0 GPUs.")
        num_available_gpus = 0 # Ensure it's 0 if check fails

    if NUM_FOLDS == 0:
        logging.info("MainProcess: NUM_FOLDS is 0. No training will be performed.")
        return

    if num_available_gpus == 0:
        logging.warning("MainProcess: No GPUs available. Assigning all folds to run on CPU.")
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
        logging.info(f"MainProcess:     Fold {i+1} (0-indexed: {i}) --> GPU {assigned_gpu_for_log}")

    training_processes = []
    for i in range(NUM_FOLDS):
        fold_idx_0_based = i
        assigned_gpu = gpu_assignment_for_folds[i]
        process_gpu_name_part = str(assigned_gpu) if assigned_gpu is not None else 'CPU'
        process_name = f"Fold-{fold_idx_0_based+1}_GPU-{process_gpu_name_part}"

        logging.info(f"MainProcess: Preparing to launch training process {process_name} for Fold {fold_idx_0_based+1} on GPU {process_gpu_name_part}.")
        p = multiprocessing.Process(
            target=train_single_fold,
            args=(fold_idx_0_based, assigned_gpu),
            name=process_name
        )
        training_processes.append(p)

    logging.info(f"MainProcess: Starting {len(training_processes)} Co-DETR training processes...")
    for p in training_processes:
        p.start()
        logging.info(f"MainProcess: Process {p.name} (PID: {p.pid}) has been started.")

    logging.info("MainProcess: All training processes have been launched. Waiting for them to complete...")
    all_folds_successful = True
    for i, p in enumerate(training_processes):
        p.join()
        logging.info(f"MainProcess: Process {p.name} (PID: {p.pid}) for Fold {i+1} has completed. Exit code: {p.exitcode}.")
        if p.exitcode != 0:
            all_folds_successful = False
            logging.error(f"MainProcess: Process {p.name} for Fold {i+1} (PID: {p.pid}) exited with a non-zero code ({p.exitcode}). "
                          f"Check its dedicated log files in {os.path.join(BASE_RESULTS_DIR, f'fold_{i+1}')} and the main orchestration error log.")

    if all_folds_successful:
        logging.info(f"MainProcess: All {NUM_FOLDS}-fold Co-DETR cross-validation training tasks have finished successfully.")
    else:
        logging.error(f"MainProcess: One or more Co-DETR training folds failed. Please review the logs.")
    logging.info(f"MainProcess: Please check individual fold directories under {BASE_RESULTS_DIR} for detailed logs and training results.")

if __name__ == '__main__':
    # Ensure this script is run from the root of the Co-DETR repository.
    # Example: python path_to_this_script/your_script_name.py

    # Pre-run checks:
    # 1. Verify `BASE_DATA_PATH_TEMPLATE` points to your COCO-style data folds.
    # 2. Verify `CODETR_BASE_CONFIG_PATH` is a valid path to a Co-DETR config file.
    # 3. Adjust `TRAIN_HYPERPARAMS` as needed, especially 'batch' size for your GPU memory.
    # 4. Ensure MMDetection and its dependencies are installed in the environment.
    # 5. Ensure `tools/train.py` is executable and present.
    run_cross_validation_training()