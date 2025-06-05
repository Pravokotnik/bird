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

target_ids = sorted(class_id_to_name.keys())    # ex: [2, 12, 14, …, 191]
orig_to_new = {orig_id: idx for idx, orig_id in enumerate(target_ids)}
new_to_orig = {idx: orig_id for orig_id, idx in orig_to_new.items()}
new_index_to_name = {idx: class_id_to_name[orig_id] for idx, orig_id in new_to_orig.items()}

# === 3. Load model and checkpoint ===
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# === 4. Preprocessing transforms ===
preprocess = T.Compose([
    T.Resize(int(input_size * 1.14)),         # shorter side ≈ 256
    T.CenterCrop(input_size),                 # center-crop 224×224
    T.ToTensor(),                             # convert to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet means
                std=[0.229, 0.224, 0.225])     # ImageNet stds
])

# === 5. Helper: Recursively collect all image paths under images_root ===
def get_all_image_paths(root_dir):
    """
    Recursively collect all image file paths (.jpg, .jpeg, .png, .bmp) under root_dir.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for dirpath, dirnames, filenam
