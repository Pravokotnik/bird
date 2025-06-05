import os
import argparse
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

def clean_class_name(folder_name: str) -> str:
    if "." in folder_name:
        return folder_name.split(".", 1)[1]
    return folder_name

class CleanNameImageFolder(datasets.ImageFolder):
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

def get_data_loaders(
    images_root: str,
    input_size: int,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4
):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = CleanNameImageFolder(root=images_root, transform=train_transforms)
    total_images = len(full_dataset)
    indices = list(range(total_images))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(total_images * val_split)
    train_indices = indices[val_size:]
    val_indices   = indices[:val_size]

    # Temporarily switch transform for validation subset
    full_dataset.transform = val_transforms
    val_subset   = torch.utils.data.Subset(full_dataset, val_indices)

    # Switch back to train transforms for training subset
    full_dataset.transform = train_transforms
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    idx_to_class = full_dataset.idx_to_clean_name
    return train_loader, val_loader, idx_to_class

def build_model(num_classes: int, feature_extract: bool=True, use_pretrained: bool=True) -> torch.nn.Module:
    model = models.resnet50(pretrained=use_pretrained)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    checkpoint_dir: str
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        ### Training phase ###
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc  = running_corrects / total_train
        print(f"[Epoch {epoch}/{num_epochs}]  Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}")

        ### Validation phase ###
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
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        val_epoch_loss = val_loss / total_val
        val_epoch_acc  = val_corrects / total_val
        print(f"[Epoch {epoch}/{num_epochs}]  Val   Loss: {val_epoch_loss:.4f}  Val   Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved new best model (epoch {epoch}, val_acc {val_epoch_acc:.4f})")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Best model located at: {best_model_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train bird-species classifier from CUB subset")
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Path to bird/cub_subset/images (each subfolder is e.g. 002.Laysan_Albatross)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="bird_checkpoints",
        help="Directory in which to save best_model.pth and idx_to_class.pth"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input crop size (default 224)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default 32)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for validation (default 0.2)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
        help="Number of epochs (default 15)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default 1e-4)"
    )
    parser.add_argument(
        "--feature_extract",
        action="store_true",
        help="If set, only fine-tune the final FC layer"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    model = build_model(num_classes, feature_extract=args.feature_extract, use_pretrained=True)
    model = model.to(device)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save idx→class mapping so inference can recover human-readable labels
    mapping_path = os.path.join(args.checkpoint_dir, "idx_to_class.pth")
    torch.save(idx_to_class, mapping_path)
    print(f"Saved idx→class mapping to: {mapping_path}")
