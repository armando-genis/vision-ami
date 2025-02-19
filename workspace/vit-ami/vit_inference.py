import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# If you're on a newer Transformers, you may use SegformerImageProcessor
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

import rerun as rr  # pip install rerun-sdk
from rerun import AnnotationContext, AnnotationInfo 

import time

######################
# Color Palette
######################
# For classes [0..5] (6 classes)
CLASS_COLORS = [
    (0, 0, 0),         # class 0 -> black
    (255, 0, 0),       # class 1 -> red
    (0, 255, 0),       # class 2 -> green
    (0, 0, 255),       # class 3 -> blue
    (255, 255, 0),     # class 4 -> yellow
    (255, 0, 255),     # class 5 -> magenta
]

# Optional: Class labels (names). Or just "Class 0" ...
CLASS_NAMES = [
    "Background",
    "PLATE",
    "BUSHELING",
    "P_S",
    "SHREDDED",
    "PIT_SCRAP",
]

def colorize_mask(mask_np):
    """
    Convert a (H, W) array of class IDs into a color RGB image (H, W, 3).
    mask_np: 2D numpy array with values in [0..num_classes-1].
    """
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(CLASS_COLORS)):
        color_mask[mask_np == class_id] = CLASS_COLORS[class_id]
    return color_mask

def label_contours_on_frame(frame_bgr, pred_mask, skip_class_zero=True):
    """
    For each unique class in `pred_mask`, find contours and label them on `frame_bgr`.
    If skip_class_zero=True, we ignore class_id=0 (often background).
    """
    unique_classes = np.unique(pred_mask)
    for class_id in unique_classes:
        if skip_class_zero and class_id == 0:
            continue

        # Create a binary mask for this class
        class_mask = (pred_mask == class_id).astype(np.uint8)

        # Find contours for this class region
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes or text for each contour
        for contour in contours:
            if contour.size < 5:
                # skip tiny contours
                continue
            x, y, w, h = cv2.boundingRect(contour)
            
            # Optional: draw bounding box (white)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Put text above the box
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
            cv2.putText(
                frame_bgr, label, 
                (x, max(y - 5, 10)),  # y-5 so text is slightly above box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # font scale
                (255, 255, 255), 2   # white text, thickness=2
            )


def get_contour_boxes(pred_mask, skip_class_zero=True):
    """Returns list of (class_id, x, y, w, h) for each detected contour."""
    unique_classes = np.unique(pred_mask)
    boxes = []
    for class_id in unique_classes:
        if skip_class_zero and class_id == 0:
            continue
        class_mask = (pred_mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.size < 5:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((class_id, x, y, w, h))
    return boxes

def main():

    rr.init("rerun_example_my_data", spawn=True)

    if len(sys.argv) < 2:
        print("Usage: python inference.py <input_video_path> [output_video_path]")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2] if len(sys.argv) > 2 else "output_segmented.mp4"

    # ---------------------------
    # 1) Load Model
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "segformer_model_best_mIoU"  # The folder with your saved model
    print(f"Loading model from {model_dir} ...")

    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    num_classes = model.config.num_labels
    print(f"Loaded model with {num_classes} classes.")


    annotation_info = [
        AnnotationInfo(
            id=class_id,
            label=CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}",
            color=CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)
        )
        for class_id in range(num_classes)
    ]
    rr.log("/", AnnotationContext(annotation_info))


    # Transform to match training
    input_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # ---------------------------
    # 2) Open Video
    # ---------------------------
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {input_video_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1.0 / fps if fps > 0 else 1.0/30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_video_path}")
    print(f"Output will be saved to: {output_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1.0 / fps if fps > 0 else 1.0/30  # fallback to 30 FPS if invalid


    frame_count = 0

    start_time = time.time()  # add import time at top

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1
        # Use real-time sequencing for proper playback speed
        current_time = frame_count * time_per_frame
        rr.set_time_seconds("real_time", current_time)


        # Convert BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        rr.log("video/rgb", rr.Image(frame_rgb))

        # Transform input
        input_tensor = input_transform(frame_pil).unsqueeze(0).to(device)  # (1,3,512,512)

        with torch.no_grad():
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits  # (1, num_classes, h_out, w_out)
            # Upsample
            upsampled = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy()  # (512, 512)

        # Colorize mask
        color_mask = colorize_mask(pred_mask)

        # Resize color mask to original shape
        color_mask_bigger = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        color_mask_bgr = cv2.cvtColor(color_mask_bigger, cv2.COLOR_RGB2BGR)

        # Blend
        alpha = 0.7
        blended = cv2.addWeighted(frame_bgr, 1 - alpha, color_mask_bgr, alpha, 0)

        # Optionally label bounding boxes for each class
        # We'll do it on the 'blended' frame for clarity
        pred_mask_fullsize = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        rr.log("video/segmentation", rr.SegmentationImage(pred_mask_fullsize))


        # Log bounding boxes
        contour_boxes = get_contour_boxes(pred_mask_fullsize, skip_class_zero=True)
        if contour_boxes:
            class_ids = np.array([box[0] for box in contour_boxes], dtype=np.uint8)
            rects = np.array([box[1:] for box in contour_boxes], dtype=np.float32)
            rr.log(
                    "video/boxes",
                    rr.Boxes2D(
                        array=rects,
                        array_format=rr.Box2DFormat.XYWH,
                        class_ids=class_ids
                        )
                    )

        # Put bounding boxes + text
        label_contours_on_frame(blended, pred_mask_fullsize, skip_class_zero=True)

        out.write(blended)

        if frame_count % 50 == 0:
            print(f"Processed frame {frame_count}...")

        if fps > 0:
            elapsed = time.time() - start_time
            target_time = frame_count * time_per_frame
            sleep_time = target_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    out.release()
    print("Inference complete! Saved:", output_video_path)

if __name__ == "__main__":
    main()
