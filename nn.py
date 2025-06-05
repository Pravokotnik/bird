import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split

# --- 1. Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 24
batch_size = 32
num_epochs_stage1 = 8
num_epochs_stage2 = 25
input_size = 224

# Paths
root = "cub_subset"        # Your filtered dataset folder
images_dir = os.path.join(root, "images")  # Contains 24 subfolders
cache_dir = "models"       # Where checkpoints will be saved

# --- 1.1. Build name_to_id and mapping orig_to_new ---
# Read classes.txt to get the 24 class IDs
name_to_id = {}
with open(os.path.join(root, "classes.txt"), "r") as f:
    for line in f:
        class_id_str, class_name = line.strip().split(" ", 1)
        # classes.txt now only contains the 24 entries
        name_to_id[class_name] = int(class_id_str)

# Create a sorted list of original IDs, then map to [0..23]
target_ids = sorted(name_to_id.values())  # e.g., [2,12,14,25,...,191]
orig_to_new = {orig_cls: new_idx for new_idx, orig_cls in enumerate(target_ids)}
# e.g., {2:0, 12:1, 14:2, 25:3, ... , 191:23}

# --- 1.2. Read filtered image-class labels ---
img_map = {}   # image_id -> relative path under 'images/'
labels = {}    # image_id -> original class_id

with open(os.path.join(root, "images.txt"), "r") as f:
    for line in f:
        idx, relp = line.strip().split()
        img_map[int(idx)] = relp

with open(os.path.join(root, "image_class_labels.txt"), "r") as f:
    for line in f:
        idx, cls = map(int, line.strip().split())
        labels[idx] = cls

# Build a list of (full_image_path, original_class_id)
all_items = []
for image_id, class_id in labels.items():
    relp = img_map[image_id]
    full_path = os.path.join(images_dir, relp)
    all_items.append((full_path, class_id))

# --- 1.3. Split into train/val/test (stratified) ---
paths, cls_ids = zip(*all_items)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, cls_ids, test_size=0.20, stratify=cls_ids, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
)

# --- 2. Custom Dataset with remapped labels ---
class BirdDataset(Dataset):
    def __init__(self, img_paths, img_labels, orig_to_new_map, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.orig_to_new = orig_to_new_map  # maps original class IDs → [0..23]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        orig_cls = self.img_labels[idx]
        label = self.orig_to_new[orig_cls]  # now in [0..23]
        return img, label

# --- 3. Data Augmentation / Transforms ---
train_transforms = T.Compose([
    T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize(int(input_size * 1.14)),
    T.CenterCrop(input_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# --- 4. Create Datasets & DataLoaders ---
train_ds = BirdDataset(train_paths, train_labels, orig_to_new, transform=train_transforms)
val_ds = BirdDataset(val_paths, val_labels, orig_to_new, transform=val_transforms)
test_ds = BirdDataset(test_paths, test_labels, orig_to_new, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# --- 5. Model Setup (ResNet-50 pretrained) ---
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# Stage 1: Freeze all but the final layer
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- 6. Training / Evaluation Helpers ---
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples

# --- 7. Stage 1 Training (Head Only) ---
best_val_acc = 0.0
for epoch in range(num_epochs_stage1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Stage1 Epoch {epoch+1}/{num_epochs_stage1} | "
          f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(cache_dir, "best_stage1.pth"))

# --- 8. Stage 2 Training (Full Fine-Tuning) ---
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_stage2)
scaler = torch.cuda.amp.GradScaler()

best_val_acc = 0.0
for epoch in range(num_epochs_stage2):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Stage2 Epoch {epoch+1}/{num_epochs_stage2} | "
          f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(cache_dir, "best_stage2.pth"))

# --- 9. Final Evaluation on Test Set ---
model.load_state_dict(torch.load(os.path.join(cache_dir, "best_stage2.pth")))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Accuracy: {test_acc*100:.2f}%")
