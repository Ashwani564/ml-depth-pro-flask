# processing.py
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import platform

# Import the package, and the specific load_rgb function
import depth_pro
from depth_pro import load_rgb

# --- Configuration ---
LEGEND_WIDTH = 300
CLASSES_TO_IGNORE = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Safety Vest", "Hardhat"}

# --- Device Selection ---
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif platform.system() == 'Darwin' and platform.machine() == 'arm64' and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"--- Using device: {DEVICE.upper()} ---")

# --- Model Loading (Load models once when the module is imported) ---
print("Loading YOLO model for images...")
YOLO_IMAGE_MODEL = YOLO('models/yolo_for_image.pt').to(DEVICE)
print("Loading YOLO model for videos...")
YOLO_VIDEO_MODEL = YOLO('models/yolo_for_video.pt').to(DEVICE)

print("Loading Depth Pro model and transforms...")
DEPTH_MODEL, DEPTH_TRANSFORM = depth_pro.create_model_and_transforms()
DEPTH_MODEL.to(DEVICE)
DEPTH_MODEL.eval()
print("--- All models loaded ---")


def update_progress(task_id, tasks_db, progress, status):
    """
    Helper function to update task progress *without* overwriting existing keys.
    """
    if task_id in tasks_db:
        task = tasks_db[task_id]
        task['progress'] = progress
        task['status'] = status

def process_image(input_path, output_path, task_id, tasks_db):
    """
    Processes a single image for object detection and depth estimation.
    """
    try:
        update_progress(task_id, tasks_db, 10, "Processing...")
        
        yolo_input_img = cv2.imread(input_path)
        if yolo_input_img is None:
            raise ValueError("Could not read the image.")

        img_h, img_w = yolo_input_img.shape[:2]
        
        update_progress(task_id, tasks_db, 30, "Running object detection...")
        yolo_results = YOLO_IMAGE_MODEL(yolo_input_img, device=DEVICE, verbose=False)[0]

        update_progress(task_id, tasks_db, 60, "Running depth estimation...")
        # ... (depth estimation code remains the same) ...
        image_for_depth_np, _, f_px = load_rgb(input_path)
        pil_image_for_depth = Image.fromarray(image_for_depth_np)
        if f_px is None:
            f_px = float(img_w)
            print("Focal length not found in EXIF, using image width as fallback.")
        f_px_tensor = torch.tensor([f_px], device=DEVICE)
        depth_input_transformed = DEPTH_TRANSFORM(pil_image_for_depth).to(DEVICE)
        with torch.no_grad():
            prediction = DEPTH_MODEL.infer(depth_input_transformed, f_px=f_px_tensor)
        depth_np = prediction["depth"].squeeze().cpu().numpy()
        depth_np_resized = cv2.resize(depth_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        update_progress(task_id, tasks_db, 80, "Visualizing results...")
        # ... (visualization code remains the same) ...
        final_image = np.ones((img_h, img_w + LEGEND_WIDTH, 3), dtype=np.uint8) * 255
        final_image[0:img_h, 0:img_w] = yolo_input_img
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        classes = yolo_results.boxes.cls.cpu().numpy()
        detection_id = 1
        for box, cls_idx in zip(boxes, classes):
            class_name = yolo_results.names[int(cls_idx)]
            if class_name in CLASSES_TO_IGNORE:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (22, 255, 133), 2)
            center_x = np.clip((x1 + x2) // 2, 0, img_w - 1)
            center_y = np.clip((y1 + y2) // 2, 0, img_h - 1)
            depth_value = depth_np_resized[center_y, center_x]
            cv2.putText(final_image, str(detection_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 191), 2)
            legend_text = f"{detection_id}: {class_name} - {depth_value:.2f}m"
            cv2.putText(final_image, legend_text, (img_w + 15, 50 + (detection_id - 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            detection_id += 1

        cv2.imwrite(output_path, final_image)
        
        # --- START: NEW DURATION CALCULATION ---
        task = tasks_db.get(task_id)
        if task and 'start_time' in task:
            task['duration'] = time.time() - task['start_time']
        # --- END: NEW DURATION CALCULATION ---

        update_progress(task_id, tasks_db, 100, "complete")
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        # Also record duration on error
        task = tasks_db.get(task_id)
        if task and 'start_time' in task:
            task['duration'] = time.time() - task['start_time']
        update_progress(task_id, tasks_db, 100, f"error: {e}")

def process_video(input_path, output_path, task_id, tasks_db):
    """
    Processes a video, applying detection and depth estimation.
    """
    try:
        # ... (video setup code remains the same) ...
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (original_w + LEGEND_WIDTH, original_h))
        FRAME_SKIP = 5
        PROCESS_WIDTH = 854
        frame_count = 0
        
        while True:
            # ... (the main frame processing loop remains the same) ...
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            if frame_count % FRAME_SKIP == 0:
                update_progress(task_id, tasks_db, progress, f"Processing frame {frame_count}/{total_frames}")
            if frame_count % FRAME_SKIP != 0:
                continue
            aspect_ratio = frame.shape[0] / frame.shape[1]
            new_h = int(PROCESS_WIDTH * aspect_ratio)
            processing_frame = cv2.resize(frame, (PROCESS_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            img_h, img_w = processing_frame.shape[:2]
            yolo_results = YOLO_VIDEO_MODEL(processing_frame, device=DEVICE, verbose=False)[0]
            rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            depth_input_transformed = DEPTH_TRANSFORM(pil_image).to(DEVICE)
            f_px_tensor = torch.tensor([float(img_w)], device=DEVICE)
            with torch.no_grad():
                prediction = DEPTH_MODEL.infer(depth_input_transformed, f_px=f_px_tensor)
            depth_np = prediction["depth"].squeeze().cpu().numpy()
            depth_np_resized = cv2.resize(depth_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            final_image_small = np.ones((img_h, img_w + LEGEND_WIDTH, 3), dtype=np.uint8) * 255
            final_image_small[0:img_h, 0:img_w] = processing_frame
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            classes = yolo_results.boxes.cls.cpu().numpy()
            detection_id = 1
            for box, cls_idx in zip(boxes, classes):
                class_name = yolo_results.names[int(cls_idx)]
                if class_name in CLASSES_TO_IGNORE: continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(final_image_small, (x1, y1), (x2, y2), (22, 255, 133), 2)
                center_x = np.clip((x1 + x2) // 2, 0, img_w - 1)
                center_y = np.clip((y1 + y2) // 2, 0, img_h - 1)
                depth_value = depth_np_resized[center_y, center_x]
                cv2.putText(final_image_small, str(detection_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 191), 2)
                legend_text = f"{detection_id}: {class_name} - {depth_value:.2f}m"
                cv2.putText(final_image_small, legend_text, (img_w + 10, 50 + (detection_id - 1) * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                detection_id += 1
            output_frame = cv2.resize(final_image_small, (original_w + LEGEND_WIDTH, original_h))
            writer.write(output_frame)

        cap.release()
        writer.release()

        # --- START: NEW DURATION CALCULATION ---
        task = tasks_db.get(task_id)
        if task and 'start_time' in task:
            task['duration'] = time.time() - task['start_time']
        # --- END: NEW DURATION CALCULATION ---
        
        update_progress(task_id, tasks_db, 100, "complete")

    except Exception as e:
        print(f"Error processing video {input_path}: {e}")
        # Also record duration on error
        task = tasks_db.get(task_id)
        if task and 'start_time' in task:
            task['duration'] = time.time() - task['start_time']
        update_progress(task_id, tasks_db, 100, f"error: {e}")