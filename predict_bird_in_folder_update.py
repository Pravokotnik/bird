# predict_birds_in_folder.py

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import numpy as np

# ─── 1. Relative paths (assumes you run this from the `bird/` folder) ─────────────
CHECKPOINT_PATH = os.path.join("bird_checkpoints_updated", "best_model_updated.pth")
MAPPING_PATH    = os.path.join("bird_checkpoints_updated", "idx_to_class_updated.pth")
IMAGES_DIR      = "images"
INPUT_SIZE      = 224
TOPK            = 3  # we'll output top-3

def load_model_and_mapping(checkpoint_path: str, mapping_path: str, device: torch.device):
    idx_to_class = torch.load(mapping_path)  # e.g. {0: "Laysan_Albatross", …}
    num_classes = len(idx_to_class)

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, idx_to_class

def crop_bird_region(img_bgr):
    """
    1) Convert to HSV, threshold for blue to isolate the ring
    2) Find largest contour, fit min enclosing circle (cx, cy, r)
    3) Crop a box just above the ring: left/right = cx ± 0.8r; top = cy - 1.5r; bottom = cy - 0.2r
    4) If any step fails, return None (so caller can fallback)
    """
    height, width = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Blue-range in HSV (tweak if needed)
    lower_blue = np.array([100, 120,  60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(largest)
    cx, cy, r = float(cx), float(cy), float(r)
    if r < 10 or r > min(width, height) / 2:
        return None

    left   = int(cx - 0.8 * r)
    right  = int(cx + 0.8 * r)
    top    = int(cy - 1.5 * r)
    bottom = int(cy - 0.2 * r)

    left   = max(0, left)
    right  = min(width, right)
    top    = max(0, top)
    bottom = min(height, bottom)
    if left >= right or top >= bottom:
        return None

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))

def top_center_crop(img_bgr):
    """224×224 crop centered horizontally, ¼ down from top."""
    height, width = img_bgr.shape[:2]
    crop_size = 224
    cx = width // 2
    cy = height // 4
    left   = max(0, cx - crop_size // 2)
    right  = min(width, cx + crop_size // 2)
    top    = max(0, cy - crop_size // 2)
    bottom = min(height, top + crop_size)
    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)
    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))

def center_crop_full(img_bgr):
    """224×224 center crop of the entire frame."""
    height, width = img_bgr.shape[:2]
    crop_size = 224
    left   = max(0, width // 2 - crop_size // 2)
    top    = max(0, height // 2 - crop_size // 2)
    right  = min(width, left + crop_size)
    bottom = min(height, top + crop_size)
    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)
    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))

# Preprocessing: resize→center crop→normalize
preproc = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.14)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_topk(model, idx_to_class, pil_crop, topk, device):
    """
    Return a list of (species, probability) for the top-k predictions on pil_crop.
    """
    tensor = preproc(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
    top_probs, top_idxs = probs.topk(topk, dim=1)
    top_probs = top_probs.cpu().squeeze(0)
    top_idxs  = top_idxs.cpu().squeeze(0)

    results = []
    for idx, p in zip(top_idxs, top_probs):
        species = idx_to_class[int(idx)]
        results.append((species, float(p.item())))
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"Error: model checkpoint not found at:\n  {CHECKPOINT_PATH}")
        exit(1)
    if not os.path.isfile(MAPPING_PATH):
        print(f"Error: idx_to_class mapping not found at:\n  {MAPPING_PATH}")
        exit(1)

    model, idx_to_class = load_model_and_mapping(CHECKPOINT_PATH, MAPPING_PATH, device)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for dp, _, fnames in os.walk(IMAGES_DIR):
        for fname in fnames:
            if os.path.splitext(fname.lower())[1] in exts:
                image_paths.append(os.path.join(dp, fname))
    image_paths.sort()

    if not image_paths:
        print(f"No images found under {IMAGES_DIR}. Exiting.")
        exit(0)

    for img_path in image_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        # 1. Generate candidate crops
        crops = []
        hsv_crop = crop_bird_region(bgr)
        if hsv_crop is not None:
            crops.append(hsv_crop)

        # Fallbacks
        crops.append(top_center_crop(bgr))
        crops.append(center_crop_full(bgr))

        # 2. Choose the crop with highest top-1 confidence
        best_conf = -1.0
        best_crop = None
        for crop in crops:
            top1 = predict_topk(model, idx_to_class, crop, 1, device)[0]  # (species, prob)
            if top1[1] > best_conf:
                best_conf = top1[1]
                best_crop = crop

        # 3. Now get top-3 predictions on best_crop
        top3 = predict_topk(model, idx_to_class, best_crop, TOPK, device)

        # 4. Print results
        print(f"\nImage: {img_path}")
        for rank, (species, prob) in enumerate(top3, start=1):
            print(f"  {rank}. {species:<30s} {prob * 100:5.2f}%")
