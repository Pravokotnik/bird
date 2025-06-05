import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image, UnidentifiedImageError

# === 1. Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 24
input_size = 224

# Paths
checkpoint_path = "models/best_stage2.pth"       # Your fine-tuned ResNet-50 checkpoint
classes_txt    = "cub_subset/classes.txt"        # Filtered 24-class CUB file
images_root    = "images"                        # Folder full of new bird photos

# === 2. Build class mappings (orig ID ↔ new idx ↔ class name) ===
class_id_to_name = {}
with open(classes_txt, "r") as f:
    for line in f:
        cid_str, cname = line.strip().split(" ", 1)
        class_id_to_name[int(cid_str)] = cname
# :contentReference[oaicite:37]{index=37}

target_ids = sorted(class_id_to_name.keys())    # [2,12,14,...,191]
orig_to_new = {orig_id: idx for idx, orig_id in enumerate(target_ids)}
new_to_orig = {idx: orig_id for orig_id, idx in orig_to_new.items()}
new_index_to_name = {idx: class_id_to_name[orig_id] for idx, orig_id in new_to_orig.items()}
# 

# === 3. Load model and checkpoint ===
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()
# :contentReference[oaicite:39]{index=39}

# === 4. Preprocessing transforms ===
preprocess = T.Compose([
    T.Resize(int(input_size * 1.14)),         # shorter side ≈ 256
    T.CenterCrop(input_size),                 # center-crop 224×224
    T.ToTensor(),                             # convert to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet means
                std=[0.229, 0.224, 0.225])     # ImageNet stds
])
# :contentReference[oaicite:40]{index=40}

# === 5. Helper: Recursively collect all image paths under images_root ===
def get_all_image_paths(root_dir):
    """
    Recursively collect all image file paths (.jpg, .jpeg, .png, .bmp) under root_dir.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in exts:
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths
# :contentReference[oaicite:41]{index=41}

all_images = get_all_image_paths(images_root)
print(f"Found {len(all_images)} total images under '{images_root}'.\n")

# === 6. Inference function for a single image ===
def predict_bird_from_path(image_path, model, preprocess, new_index_to_name, device):
    """
    Given an image file path, returns (species_name, confidence).
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError):
        return None, None

    img_tensor = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
    with torch.no_grad():
        logits = model(img_tensor)                         # [1,24]
        probs = F.softmax(logits, dim=1)                    # [1,24]
        top_prob, top_idx = probs.topk(1, dim=1)            # Both are [1,1]
    pred_idx = top_idx.item()                               # 0..23
    confidence = top_prob.item()                            # 0..1
    species_name = new_index_to_name[pred_idx]              # e.g. "073.Blue_Jay"
    return species_name, confidence
# :contentReference[oaicite:42]{index=42}

# === 7. Loop over all images and print predictions ===
print("Classifying all images...\n")
for img_path in all_images:
    species, conf = predict_bird_from_path(img_path, model, preprocess, new_index_to_name, device)
    if species is None:
        continue  # skip invalid files

    # Print: <relative_path> → <species> (confidence%)
    rel_path = os.path.relpath(img_path, images_root)
    print(f"{rel_path}  →  {species}  ({conf*100:.1f}%)")
