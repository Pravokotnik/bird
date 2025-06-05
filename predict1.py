import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image, ImageDraw
from ultralytics import YOLO

# 1) Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 24
input_size = 224

# Paths
checkpoint_path = "models/best_stage2.pth"    # Your fine-tuned ResNet-50
classes_txt     = "cub_subset/classes.txt"    # 24 classes file
images_root     = "images"                    # Folder of new bird photos
vis_root        = "vis"                       # Where to save annotated images

# 2) Build class mappings (orig ID → new index → class name)
class_id_to_name = {}
with open(classes_txt, "r") as f:
    for line in f:
        cid_str, cname = line.strip().split(" ", 1)
        class_id_to_name[int(cid_str)] = cname
# :contentReference[oaicite:10]{index=10}

target_ids = sorted(class_id_to_name.keys())     # e.g. [2, 12, 14, …, 191]
orig_to_new   = {orig: idx for idx, orig in enumerate(target_ids)}
new_to_orig   = {idx: orig for orig, idx in orig_to_new.items()}
new_index_to_name = {idx: class_id_to_name[orig_id] for idx, orig_id in new_to_orig.items()}
# :contentReference[oaicite:11]{index=11}

# 3) Load fine-tuned ResNet-50 (24 classes)
model_resnet = models.resnet50(pretrained=False)
num_features = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_features, num_classes)
model_resnet.load_state_dict(torch.load(checkpoint_path, map_location=device))
model_resnet = model_resnet.to(device)
model_resnet.eval()
# :contentReference[oaicite:12]{index=12}

# 4) Load YOLOv5 (COCO pretrained)

model_yolo = YOLO("yolov5s.pt")
# :contentReference[oaicite:13]{index=13}

# 5) Set YOLO thresholds
model_yolo.conf = 0.30  # confidence threshold
model_yolo.iou  = 0.45  # IoU threshold for NMS
# :contentReference[oaicite:14]{index=14}

# 6) Preprocessing transforms for ResNet
preprocess_resnet = T.Compose([
    T.Resize(int(input_size * 1.14)),         # shorter side ≈ 256
    T.CenterCrop(input_size),                 # center-crop 224×224
    T.ToTensor(),                             # [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet means
                std=[0.229, 0.224, 0.225])     # ImageNet stds
])
# :contentReference[oaicite:15]{index=15}

# 7) Helper: Gather all image paths under images_root
def get_all_image_paths(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in exts:
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths
# :contentReference[oaicite:16]{index=16}

all_images = get_all_image_paths(images_root)
print(f"Found {len(all_images)} images under '{images_root}'.\n")

# 8) Functions to detect, crop, classify, and visualize
def detect_bird_bbox(image_path):
    """
    Returns a list of (x1, y1, x2, y2, confidence) for each detected bird.
    """
    # Run inference by calling the YOLOv5 model directly
    results = model_yolo(image_path)  # uses model_yolo.conf & model_yolo.iou
    detections = results.pandas().xyxy[0]  # Pandas DataFrame
    bird_boxes = []
    for _, row in detections.iterrows():
        if row["name"] == "bird":  # COCO class 'bird' → index 14
            x1, y1 = int(row["xmin"]), int(row["ymin"])
            x2, y2 = int(row["xmax"]), int(row["ymax"])
            conf = float(row["confidence"])
            bird_boxes.append((x1, y1, x2, y2, conf))
    return bird_boxes
# :contentReference[oaicite:17]{index=17}

def crop_bird(image_path, bbox, pad=10):
    """
    Crops the image to the bounding box (x1, y1, x2, y2) with padding.
    Returns a PIL.Image.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    x1, y1, x2, y2 = bbox
    # Add padding (clamped to image boundaries)
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(width, x2 + pad)
    y2p = min(height, y2 + pad)
    return img.crop((x1p, y1p, x2p, y2p))
# :contentReference[oaicite:18]{index=18}

def classify_cropped_bird(cropped_img):
    """
    Takes a PIL.Image (cropped bird), preprocesses it, runs ResNet inference,
    and returns (species_name, confidence).
    """
    img_tensor = preprocess_resnet(cropped_img).unsqueeze(0).to(device)  # [1,3,224,224]
    with torch.no_grad():
        logits = model_resnet(img_tensor)       # [1,24]
        probs  = F.softmax(logits, dim=1)       # [1,24]
        top_prob, top_idx = probs.topk(1, dim=1)# [1,1]
    pred_idx   = top_idx.item()                  # integer 0..23
    confidence = top_prob.item()                 # float 0..1
    species    = new_index_to_name[pred_idx]     # e.g. "073.Blue_Jay"
    return species, confidence
# :contentReference[oaicite:19]{index=19}

from PIL import Image, ImageDraw, ImageFont

def draw_bbox_and_label(image_path, bbox, species, confidence, output_path):
    """
    Draws a red bounding box and label (species, confidence) on the original image,
    then saves to output_path.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = bbox

    # 1) Draw bounding box (red, width=3)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # 2) Compute text size using textbbox (Pillow ≥ 10.0)
    text = f"{species} {confidence*100:.1f}%"
    font = ImageFont.load_default()  # or specify a TTF: ImageFont.truetype("arial.ttf", size=14)

    # get bounding box of the text at (0, 0)
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]
    # :contentReference[oaicite:9]{index=9}

    # 3) Draw filled red background rectangle for the text
    #    Position the label so its bottom-left corner is at (x1, y1)
    draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")

    # 4) Draw the white text on top
    draw.text((x1, y1 - text_h), text, font=font, fill="white")

    # 5) Save annotated image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
# :contentReference[oaicite:20]{index=20}

# 9) Main loop: Detect → Crop → Classify → Visualize
os.makedirs(vis_root, exist_ok=True)

print("Processing all images:\n")
for img_path in all_images:
    bird_boxes = detect_bird_bbox(img_path)
    if not bird_boxes:
        print(f"[No Bird Detected] {img_path}")
        continue

    # Pick the highest-confidence YOLO bird box
    best_bbox = max(bird_boxes, key=lambda b: b[4])  # (x1,y1,x2,y2,conf)
    x1, y1, x2, y2, y_conf = best_bbox

    # Crop that region from the original image
    cropped_img = crop_bird(img_path, (x1, y1, x2, y2), pad=10)

    # Classify with ResNet
    species, c_conf = classify_cropped_bird(cropped_img)
    print(f"{img_path}  →  {species}  ({c_conf*100:.1f}%)")

    # Save a visualization
    rel_path = os.path.relpath(img_path, images_root)
    out_vis  = os.path.join(vis_root, rel_path)
    draw_bbox_and_label(img_path, (x1, y1, x2, y2), species, c_conf, out_vis)
    print(f"  [Visualization saved to {out_vis}]\n")
