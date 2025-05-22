import os
import json
import logging
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import argparse

# Suppress excessive warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Script Configuration ---

# --- Paths ---
DEFAULT_DATASET_FOLDER = "/blue/hulcr/gmarais/PhD/phase_1_data/0_20250515/data/" # Example path
OUTPUT_DIRECTORY = os.getcwd()
LOG_FILE_NAME = "grounding_dino_evaluation_report.txt"
LOG_FILE_PATH = os.path.join(OUTPUT_DIRECTORY, LOG_FILE_NAME)

# --- Model & Processing Hyperparameters ---
MODEL_ID = "IDEA-Research/grounding-dino-base"
TEXT_PROMPT = "insect." # This will be treated as the single class for AP/mAP
BOX_THRESHOLD = 0.4 # Confidence threshold for DINO's initial predictions
TEXT_THRESHOLD = 0.3 # Another DINO threshold

# --- Evaluation Hyperparameters ---
# This IOU_THRESHOLD is used for the basic P/R/F1 and the specific mAP@0.5 calculation
PRIMARY_IOU_THRESHOLD = 0.5

# --- Other Configurations ---
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
Image.MAX_IMAGE_PIXELS = None


def setup_logging():
    """Sets up logging to both file and console."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w'),
            logging.StreamHandler()
        ]
    )

def calculate_iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def load_ground_truth_boxes(json_path, image_id):
    """Load ground truth bounding boxes from a JSON file."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        gt_boxes_details = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") == "rectangle":
                points = shape["points"]
                x_coords = sorted([points[0][0], points[1][0]])
                y_coords = sorted([points[0][1], points[1][1]])
                gt_boxes_details.append({
                    "bbox": [x_coords[0], y_coords[0], x_coords[1], y_coords[1]],
                    "image_id": image_id,
                    "used": False # For AP calculation matching
                })
        return gt_boxes_details
    except Exception as e:
        logging.error(f"Error loading JSON {json_path}: {e}")
        return []

def get_dino_predictions(image_path, model, processor, device, current_text_prompt, image_pil,
                         box_thresh_param, text_thresh_param):
    """Get bounding box predictions and scores from GroundingDINO."""
    try:
        inputs = processor(images=image_pil, text=current_text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh_param,
            text_threshold=text_thresh_param,
            target_sizes=target_sizes
        )
        pred_boxes = results[0]['boxes'].cpu().tolist() if results[0]['boxes'].numel() > 0 else []
        pred_scores = results[0]['scores'].cpu().tolist() if results[0]['scores'].numel() > 0 else []
        return pred_boxes, pred_scores
    except Exception as e:
        logging.error(f"Error during DINO prediction for {image_path}: {e}")
        return [], []

def match_boxes_and_count_metrics_for_image(pred_boxes, gt_boxes_current_image_details, current_iou_threshold):
    """
    Matches predicted boxes to ground truth boxes for a single image for P/R/F1.
    """
    tp = 0
    num_gt = len(gt_boxes_current_image_details)
    num_pred = len(pred_boxes)

    if num_gt == 0: return 0, num_pred, 0
    if num_pred == 0: return 0, 0, num_gt

    gt_bboxes_for_matching = [item["bbox"] for item in gt_boxes_current_image_details]
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt_box_coords in enumerate(gt_bboxes_for_matching):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt_box_coords, pred_box)

    gt_matched_flags = [False] * num_gt
    pred_matched_flags = [False] * num_pred
    potential_matches = []
    for gt_idx in range(num_gt):
        for pred_idx in range(num_pred):
            if iou_matrix[gt_idx, pred_idx] >= current_iou_threshold:
                potential_matches.append((iou_matrix[gt_idx, pred_idx], gt_idx, pred_idx))
    potential_matches.sort(key=lambda x: x[0], reverse=True)

    for _, gt_idx, pred_idx in potential_matches:
        if not gt_matched_flags[gt_idx] and not pred_matched_flags[pred_idx]:
            tp += 1
            gt_matched_flags[gt_idx] = True
            pred_matched_flags[pred_idx] = True
            
    fn = num_gt - sum(gt_matched_flags)
    fp = num_pred - sum(pred_matched_flags)
    return tp, fp, fn

def calculate_average_precision(all_predictions_details, all_ground_truths_details, iou_threshold_for_ap_calc):
    """
    Calculates Average Precision (AP) using COCO-style interpolation for a specific IoU threshold.
    """
    if not all_predictions_details:
        return 0.0, []

    # Crucially, reset 'used' state for GTs for this specific AP calculation pass
    # This allows this function to be called multiple times with different IoU thresholds
    # on the same master list of GTs.
    temp_ground_truths_details = {}
    for img_id, gts in all_ground_truths_details.items():
        temp_ground_truths_details[img_id] = []
        for gt in gts:
            temp_ground_truths_details[img_id].append(gt.copy()) # Create a copy
            temp_ground_truths_details[img_id][-1]['used'] = False # Ensure 'used' is reset

    current_predictions_details = sorted(all_predictions_details, key=lambda x: x['score'], reverse=True)
    num_total_gt = sum(len(gts) for gts in temp_ground_truths_details.values())

    if num_total_gt == 0:
        return 0.0, []

    tp_arr = np.zeros(len(current_predictions_details))
    fp_arr = np.zeros(len(current_predictions_details))

    for i, pred in enumerate(current_predictions_details):
        gt_for_image = temp_ground_truths_details.get(pred['image_id'], [])
        best_iou = -1
        best_gt_idx = -1

        for j, gt in enumerate(gt_for_image):
            if not gt['used']: # Only consider unused GT boxes
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold_for_ap_calc and best_gt_idx != -1: # Check best_gt_idx was found
            # Only mark as TP if the best_gt_idx was indeed for an unused GT box
            # (This check is slightly redundant due to 'if not gt['used']' above but safe)
            if not gt_for_image[best_gt_idx]['used']:
                 tp_arr[i] = 1
                 gt_for_image[best_gt_idx]['used'] = True
            else: # Should not happen if logic is correct
                 fp_arr[i] = 1
        else:
            fp_arr[i] = 1

    acc_tp = np.cumsum(tp_arr)
    acc_fp = np.cumsum(fp_arr)

    # Handle cases where acc_tp + acc_fp might be zero (no predictions or all zero scores initially)
    precision_pts = np.zeros_like(acc_tp, dtype=float)
    valid_mask = (acc_tp + acc_fp) > 0
    precision_pts[valid_mask] = acc_tp[valid_mask] / (acc_tp[valid_mask] + acc_fp[valid_mask])
    
    recall_pts = acc_tp / num_total_gt if num_total_gt > 0 else np.zeros_like(acc_tp, dtype=float)

    mrec = np.concatenate(([0.], recall_pts, [1.]))
    mpre = np.concatenate(([1.], precision_pts, [0.])) # Start P=1 at R=0, end P=0 at R=1. Some use 0 for start.

    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i+1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    
    unique_recalls, unique_indices = np.unique(mrec, return_index=True)
    pr_curve_points = sorted(list(zip(unique_recalls, mpre[unique_indices])), key=lambda x: x[0])

    return ap, pr_curve_points


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="GroundingDINO Evaluation Script")
    parser.add_argument("--dataset_folder", type=str, default=DEFAULT_DATASET_FOLDER, help=f"Path to dataset. Default: {DEFAULT_DATASET_FOLDER}")
    args = parser.parse_args()
    dataset_folder = args.dataset_folder

    try:
        logging.info("Starting GroundingDINO evaluation script.")
        logging.info(f"Report will be saved to: {LOG_FILE_PATH}")
        logging.info(f"--- Current Configuration ---")
        logging.info(f"Dataset Folder: {dataset_folder}")
        logging.info(f"Model ID: {MODEL_ID}")
        logging.info(f"Text Prompt (Class): '{TEXT_PROMPT}'")
        logging.info(f"Box Threshold (DINO confidence): {BOX_THRESHOLD}")
        logging.info(f"Text Threshold (DINO): {TEXT_THRESHOLD}")
        logging.info(f"Primary IoU Threshold (for P/R/F1 & mAP@0.5): {PRIMARY_IOU_THRESHOLD}")
        logging.info(f"-----------------------------")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {DEVICE}")

        if not os.path.isdir(dataset_folder): logging.error(f"Dataset folder not found: {dataset_folder}"); return
        logging.info(f"Loading GroundingDINO model: {MODEL_ID}...")
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE); model.eval()
        except Exception as e: logging.error(f"Failed to load model: {e}", exc_info=True); return
        logging.info("Model loaded successfully.")

        total_images_processed = 0; total_images_with_gt = 0; total_images_without_gt = 0
        total_images_with_dino_pred = 0; total_images_without_dino_pred = 0
        accumulated_gt_boxes_count = 0; accumulated_dino_boxes_count = 0
        img_level_tp = 0; img_level_fp = 0; img_level_fn = 0
        all_predictions_for_ap = []; all_ground_truths_for_ap = {}
        images_dino_pred_but_gt_missing_or_empty = 0; total_dino_boxes_in_images_without_gt = 0
        images_gt_exists_but_dino_no_pred = 0; dino_predictions_in_gt_images_count = 0

        logging.info(f"Starting to process images in: {dataset_folder}")
        image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not image_files: logging.warning(f"No images found in {dataset_folder}"); return

        for filename in image_files:
            total_images_processed += 1
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(dataset_folder, filename)
            json_path = os.path.join(dataset_folder, image_id + ".json")
            logging.info(f"Processing image ({total_images_processed}/{len(image_files)}): {filename}")
            try: image_pil = Image.open(image_path).convert("RGB")
            except Exception as e: logging.error(f"Could not open image {image_path}: {e}"); continue

            gt_boxes_current_image_details = load_ground_truth_boxes(json_path, image_id)
            num_gt_boxes_current_image = len(gt_boxes_current_image_details)
            accumulated_gt_boxes_count += num_gt_boxes_current_image
            has_gt = num_gt_boxes_current_image > 0
            if has_gt: total_images_with_gt += 1; all_ground_truths_for_ap[image_id] = gt_boxes_current_image_details
            else: total_images_without_gt += 1

            pred_boxes_coords, pred_scores = get_dino_predictions(image_path, model, processor, DEVICE, TEXT_PROMPT, image_pil, BOX_THRESHOLD, TEXT_THRESHOLD)
            num_pred_boxes_current_image = len(pred_boxes_coords)
            accumulated_dino_boxes_count += num_pred_boxes_current_image
            if num_pred_boxes_current_image > 0: total_images_with_dino_pred += 1
            else: total_images_without_dino_pred += 1

            if has_gt:
                dino_predictions_in_gt_images_count += num_pred_boxes_current_image
                tp_img, fp_img, fn_img = match_boxes_and_count_metrics_for_image(pred_boxes_coords, gt_boxes_current_image_details, PRIMARY_IOU_THRESHOLD)
                img_level_tp += tp_img; img_level_fp += fp_img; img_level_fn += fn_img
                if num_pred_boxes_current_image == 0: images_gt_exists_but_dino_no_pred +=1
                for i in range(num_pred_boxes_current_image):
                    all_predictions_for_ap.append({"image_id": image_id, "bbox": pred_boxes_coords[i], "score": pred_scores[i]})
            else:
                if num_pred_boxes_current_image > 0:
                    images_dino_pred_but_gt_missing_or_empty += 1
                    total_dino_boxes_in_images_without_gt += num_pred_boxes_current_image
        
        logging.info("\n" + "="*60 + "\n--- Overall Dataset Report ---\n" + "="*60)
        logging.info(f"Dataset Folder Used: {dataset_folder}\nText Prompt: '{TEXT_PROMPT}'\nModel ID: {MODEL_ID}")
        logging.info(f"DINO Box Thr: {BOX_THRESHOLD}, DINO Text Thr: {TEXT_THRESHOLD}\n" + "-"*40)
        logging.info(f"Total images processed: {total_images_processed}")
        logging.info(f"  Images with GT: {total_images_with_gt}\n  Images without GT: {total_images_without_gt}\n" + "-"*40)
        logging.info(f"Total images DINO made predictions: {total_images_with_dino_pred}")
        logging.info(f"Total images DINO NO predictions: {total_images_without_dino_pred}\n" + "-"*40)
        logging.info(f"Total GT boxes (JSONs): {accumulated_gt_boxes_count}")
        logging.info(f"Total DINO boxes (all images): {accumulated_dino_boxes_count}\n" + "-"*40)
        logging.info("Scenario Counts:")
        logging.info(f"  Images DINO pred BUT no GT: {images_dino_pred_but_gt_missing_or_empty}")
        logging.info(f"    - Total DINO boxes in these 'unannotated' images: {total_dino_boxes_in_images_without_gt}")
        logging.info(f"  Images WITH GT BUT DINO NO pred: {images_gt_exists_but_dino_no_pred}")

        logging.info("\n" + "="*60 + "\n--- Evaluation Report for Images WITH Ground Truth Annotations ---\n" + "="*60)
        logging.info(f"Number of images with GT considered: {total_images_with_gt}")
        total_gt_in_gt_images_for_pr_f1 = img_level_tp + img_level_fn
        logging.info(f"Total GT boxes in these {total_images_with_gt} images (for P/R/F1): {total_gt_in_gt_images_for_pr_f1}")
        logging.info(f"Total DINO predictions in these {total_images_with_gt} images (passed BOX_THR): {dino_predictions_in_gt_images_count}")
        
        logging.info("-" * 40 + f"\nMetrics at fixed DINO confidence (BOX_THR={BOX_THRESHOLD}) & IoU={PRIMARY_IOU_THRESHOLD}:")
        logging.info(f"  True Positives (TP): {img_level_tp}\n  False Positives (FP): {img_level_fp}\n  False Negatives (FN): {img_level_fn}")
        precision = img_level_tp / (img_level_tp + img_level_fp) if (img_level_tp + img_level_fp) > 0 else 0.0
        recall = img_level_tp / total_gt_in_gt_images_for_pr_f1 if total_gt_in_gt_images_for_pr_f1 > 0 else 0.0
        f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
        logging.info(f"  Precision: {precision:.4f}\n  Recall: {recall:.4f}\n  F1-Score: {f1_score:.4f}")
        
        logging.info("-" * 40 + "\nMetrics across DINO confidence scores (for AP/mAP calculations):")
        num_total_gt_for_ap = sum(len(gts) for gts in all_ground_truths_for_ap.values())
        logging.info(f"  Total GT boxes for AP: {num_total_gt_for_ap}")
        logging.info(f"  Total DINO predictions for AP: {len(all_predictions_for_ap)}")

        logging.info(f"\n  --- Calculating mAP@{PRIMARY_IOU_THRESHOLD} (Primary IoU Threshold) ---")
        ap_at_primary_iou, pr_curve_primary = calculate_average_precision(all_predictions_for_ap, all_ground_truths_for_ap, PRIMARY_IOU_THRESHOLD)
        logging.info(f"    Average Precision (AP@{PRIMARY_IOU_THRESHOLD}) for '{TEXT_PROMPT}': {ap_at_primary_iou:.4f}")
        logging.info(f"    Mean Average Precision (mAP@{PRIMARY_IOU_THRESHOLD}): {ap_at_primary_iou:.4f} (single class)")
        # Optional: Log PR curve for primary IoU
        # logging.info(f"    PR Curve Points for AP@{PRIMARY_IOU_THRESHOLD}:")
        # for r,p in pr_curve_primary: logging.info(f"      R: {r:.3f}, P: {p:.3f}")

        logging.info(f"\n  --- Calculating mAP@[.5:.05:.95] (COCO-style) ---")
        coco_iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1)
        ap_per_iou_coco = []
        for coco_thresh in coco_iou_thresholds:
            logging.info(f"    Calculating AP for IoU threshold: {coco_thresh:.2f}")
            ap_val, _ = calculate_average_precision(all_predictions_for_ap, all_ground_truths_for_ap, coco_thresh)
            ap_per_iou_coco.append(ap_val)
            logging.info(f"      AP@{coco_thresh:.2f} for '{TEXT_PROMPT}': {ap_val:.4f}")
        
        mean_ap_coco = np.mean(ap_per_iou_coco) if ap_per_iou_coco else 0.0
        logging.info(f"    Individual APs for IoUs {coco_iou_thresholds.tolist()}: {[f'{apv:.4f}' for apv in ap_per_iou_coco]}")
        logging.info(f"    Mean Average Precision (mAP@[.5:.05:.95]) for '{TEXT_PROMPT}': {mean_ap_coco:.4f}")
        
        logging.info("-" * 40 + "\nNote on reports:\n  P/R/F1 are based on DINO predictions passing the fixed BOX_THRESHOLD and evaluated at a single IoU.")
        logging.info("  AP/mAP metrics consider varying confidence scores of DINO predictions (that passed BOX_THRESHOLD).")
        logging.info("  All above metrics calculated ONLY for images with ground truth annotations.")
        logging.info("\n---------DONE---------")

    except KeyboardInterrupt: logging.info("\nScript interrupted by user.")
    except Exception as e: logging.error(f"An unexpected critical error: {e}", exc_info=True)
    finally: logging.info(f"Script execution finished. Log: {LOG_FILE_PATH}")

if __name__ == '__main__':
    main()