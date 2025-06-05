# train_bird_classifier_updated.py

import os
import argparse
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ────────────────────────────────────────────────────────────────────────────────
# 1. “Clean” class names by stripping numeric prefixes
# ────────────────────────────────────────────────────────────────────────────────
def clean_class_name(folder_name: str) -> str:
    if "." in folder_name:
        return folder_name.split(".", 1)[1]
    return folder_name

class CleanNameImageFolder(datasets.ImageFolder):
    """
    Subclass of ImageFolder that strips off the numeric prefix (e.g. "002.") from each
    class‐folder and builds:
      - idx_to_clean_name: {0: "Laysan_Albatross", …}
      - class_to_idx:      {"Laysan_Albatross": 0, …}
    """
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)

        self.idx_to_clean_name = {}
        cleaned_to_idx = {}

        for folder_name, idx in self.class_to_idx.items():
            clean_name = clean_class_name(folder_name)
            cleaned_to_idx[clean_name] = idx
            self.idx_to_clean_name[idx] = clean_name

        self.classes = [self.idx_to_clean_name[i] for i in range(len(self.idx_to_clean_name))]
        self.class_to_idx = cleaned_to_idx

# ────────────────────────────────────────────────────────────────────────────────
# 2. DataLoader builder with corrected transform order (PIL→Tensor→Erase)
# ────────────────────────────────────────────────────────────────────────────────
def get_data_loaders(
    images_root: str,
    input_size: int,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4
):
    """
    1) Loads ALL images from `images_root` via CleanNameImageFolder.
    2) Applies strong PIL‐based augmentations first.
    3) Converts to Tensor → Normalize → then RandomErasing.
    4) Splits dataset into train/validation by index.
    5) Returns train_loader, val_loader, idx_to_class mapping.
    """

    # A) PIL‐based augmentations (bird‐only crops assumed)
    pil_transforms = transforms.Compose([
        # 1) RandomResizedCrop: bird can be 5%–100% of the 224×224 patch
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.05, 1.0),
            ratio=(0.8, 1.25)
        ),
        # 2) Random flips + slight rotation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        # 3) Color jitter to simulate varied lighting
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        # 4) Occasional Gaussian blur to simulate out-of-focus / small bird
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
        # Finally, convert to Tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # B) Validation transforms (PIL→Tensor→Normalize only)
    val_transforms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # C) Wrap the above plus RandomErasing into final train transforms
    train_transforms = transforms.Compose([
        pil_transforms,
        transforms.RandomErasing(
            p=0.6,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        )
    ])

    # Load the full dataset with train_transforms first
    full_dataset = CleanNameImageFolder(root=images_root, transform=train_transforms)
    total_images = len(full_dataset)
    indices = list(range(total_images))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(total_images * val_split)
    train_indices = indices[val_size:]
    val_indices   = indices[:val_size]

    # Build validation subset using val_transforms
    full_dataset.transform = val_transforms
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    # Switch back to train_transforms for the remaining dataset
    full_dataset.transform = train_transforms
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)

    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    idx_to_class = full_dataset.idx_to_clean_name
    return train_loader, val_loader, idx_to_class

# ────────────────────────────────────────────────────────────────────────────────
# 3. Build ResNet-50 (unfreeze all layers unless --feature_extract)
# ────────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, feature_extract: bool = False, use_pretrained: bool = True) -> torch.nn.Module:
    """
    Loads ResNet-50. If feature_extract=True, freezes layer1–layer3, leaving
    layer4 + fc trainable. Otherwise, all layers are trainable. Replaces fc.
    """
    model = models.resnet50(pretrained=use_pretrained)
    if feature_extract:
        for name, param in model.named_parameters():
            if not (name.startswith("layer4") or "fc" in name):
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ────────────────────────────────────────────────────────────────────────────────
# 4. Training loop with ReduceLROnPlateau & saving to a new folder/filenames
# ────────────────────────────────────────────────────────────────────────────────
def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    checkpoint_dir: str
):
    """
    - CrossEntropyLoss
    - AdamW optimizer (trainable params only)
    - ReduceLROnPlateau scheduler (on val accuracy)
    - Save best model under checkpoint_dir/best_model_updated.pth
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model_updated.pth")

    for epoch in range(1, num_epochs + 1):
        # ── Training Phase ─────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)                      # [B, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels).item()
            total_train += inputs.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc  = running_corrects / total_train
        print(f"[Epoch {epoch}/{num_epochs}]  Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}")

        # ── Validation Phase ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_corrects += torch.sum(preds == labels).item()
                total_val += inputs.size(0)

        val_epoch_loss = val_loss / total_val
        val_epoch_acc  = val_corrects / total_val
        print(f"[Epoch {epoch}/{num_epochs}]  Val   Loss: {val_epoch_loss:.4f}  Val   Acc: {val_epoch_acc:.4f}")

        # Step the scheduler on validation accuracy
        scheduler.step(val_epoch_acc)

        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved new best model (epoch {epoch}, val_acc {val_epoch_acc:.4f})")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")

# ────────────────────────────────────────────────────────────────────────────────
# 5. Argument parsing (with updated defaults and folder names)
# ────────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Retrain bird‐classifier (updated) from CUB subset")
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Path to bird/cub_subset/images (subfolders like `002.Laysan_Albatross`)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="bird_checkpoints_updated",
        help="Directory where best_model_updated.pth and idx_to_class_updated.pth will be saved"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input crop size (default: 224)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training/validation (default: 32)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data held out for validation (default: 0.2)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Base learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--feature_extract",
        action="store_true",
        help="If set, freeze layer1–layer3, train only layer4 + fc"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# 6. Main entrypoint
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1) Build data loaders
    train_loader, val_loader, idx_to_class = get_data_loaders(
        images_root=args.images_root,
        input_size=args.input_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed
    )

    num_classes = len(idx_to_class)
    print(f"→ Found {num_classes} bird species:")
    for idx, cname in idx_to_class.items():
        print(f"    {idx:02d}: {cname}")

    # 2) Build the ResNet‐50 model and move to device
    model = build_model(
        num_classes=num_classes,
        feature_extract=args.feature_extract,
        use_pretrained=True
    )
    model = model.to(device)

    # 3) Train and save best checkpoint
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    # 4) Save updated idx→class mapping under a new filename
    mapping_path = os.path.join(args.checkpoint_dir, "idx_to_class_updated.pth")
    torch.save(idx_to_class, mapping_path)
    print(f"Saved idx→class mapping to: {mapping_path}")
