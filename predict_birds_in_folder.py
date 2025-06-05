import os
import argparse
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

def get_all_image_paths(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname.lower())[1] in exts:
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths

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

def preprocess_image(image_path: str, input_size: int, device: torch.device):
    val_transforms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_t = val_transforms(img).unsqueeze(0).to(device)
    return img_t

def predict_topk_single(model: nn.Module, idx_to_class: dict, image_tensor: torch.Tensor, topk: int = 5):
    with torch.no_grad():
        logits = model(image_tensor)          # [1, num_classes]
        probs  = F.softmax(logits, dim=1)     # [1, num_classes]

    top_probs, top_idxs = probs.topk(topk, dim=1)
    top_probs = top_probs.cpu().squeeze(0)
    top_idxs  = top_idxs.cpu().squeeze(0)

    results = []
    for idx, p in zip(top_idxs, top_probs):
        species = idx_to_class[int(idx)]
        results.append((species, float(p.item())))
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict bird species for every image in a folder (e.g. birds/images)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to your trained model checkpoint (best_model.pth)."
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Path to idx_to_class mapping (idx_to_class.pth)."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing all images you want to classify (e.g. birds/images)."
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input size used during training/validation (default: 224)."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top predictions to print per image (default: 5)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model, idx_to_class = load_model_and_mapping(
        checkpoint_path=args.checkpoint,
        mapping_path=args.mapping,
        device=device
    )

    image_paths = get_all_image_paths(args.images_dir)
    if not image_paths:
        print(f"No images found in `{args.images_dir}`. Exiting.")
        exit(0)

    for img_path in sorted(image_paths):
        try:
            img_tensor = preprocess_image(img_path, args.input_size, device)
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Warning: could not open `{img_path}` → {e}. Skipping.")
            continue

        topk_results = predict_topk_single(model, idx_to_class, img_tensor, topk=args.topk)

        print(f"Image: {img_path}")
        for rank, (species, prob) in enumerate(topk_results, start=1):
            print(f"  {rank}. {species:<30s} {prob * 100:5.2f}%")
        print()
