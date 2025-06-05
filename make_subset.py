import os
import shutil
import pandas as pd

# === CONFIGURATION ===
cub_root = "CUB_200_2011"     # change to your actual path
subset_root = "cub_subset"    # where you want the subset to live

target_names = {
    "002.Laysan_Albatross",
    "012.Yellow_headed_Blackbird",
    "014.Indigo_Bunting",
    "025.Pelagic_Cormorant",
    "029.American_Crow",
    "033.Yellow_billed_Cuckoo",
    "035.Purple_Finch",
    "042.Vermilion_Flycatcher",
    "048.European_Goldfinch",
    "050.Eared_Grebe",
    "059.California_Gull",
    "068.Ruby_throated_Hummingbird",
    "073.Blue_Jay",
    "081.Pied_Kingfisher",
    "095.Baltimore_Oriole",
    "101.White_Pelican",
    "106.Horned_Puffin",
    "108.White_necked_Raven",
    "112.Great_Grey_Shrike",
    "118.House_Sparrow",
    "134.Cape_Glossy_Starling",
    "138.Tree_Swallow",
    "144.Common_Tern",
    "191.Red_headed_Woodpecker",
}

# === STEP 1: Read classes.txt and build name_to_id map ===
classes_file = os.path.join(cub_root, "classes.txt")
name_to_id = {}
with open(classes_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        class_id_str, class_name = line.split(" ", 1)
        if class_name in target_names:
            name_to_id[class_name] = int(class_id_str)

if len(name_to_id) != len(target_names):
    missing = target_names - set(name_to_id.keys())
    raise ValueError(f"Could not find these species in classes.txt: {missing}")

id_to_name = {v: k for k, v in name_to_id.items()}

# === STEP 2: Filter image_class_labels.txt by class_id ===
labels_file = os.path.join(cub_root, "image_class_labels.txt")
filtered_labels = []  # list of (image_id, class_id)
target_ids = set(name_to_id.values())

with open(labels_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        image_id_str, class_id_str = line.split()
        class_id = int(class_id_str)
        if class_id in target_ids:
            filtered_labels.append((int(image_id_str), class_id))

# Build the set of filtered image IDs
filtered_image_ids = set(img_id for (img_id, _) in filtered_labels)

# === STEP 3: Load images.txt so we can map image_id -> file path ===
images_file = os.path.join(cub_root, "images.txt")
imgid_to_path = {}
with open(images_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        image_id_str, rel_path = line.split(" ", 1)
        imgid_to_path[int(image_id_str)] = rel_path

# Sanity check: ensure all filtered image IDs exist in images.txt
for img_id in filtered_image_ids:
    if img_id not in imgid_to_path:
        raise ValueError(f"Image ID {img_id} not found in images.txt")

# === STEP 4: (Optional) Filter train_test_split.txt ===
# If you want to keep both train and test, skip this step. Otherwise:
split_file = os.path.join(cub_root, "train_test_split.txt")
imageid_is_train = {}
with open(split_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        image_id_str, is_train_str = line.split()
        imageid_is_train[int(image_id_str)] = bool(int(is_train_str))

filtered_train = [(img_id, cls_id) for (img_id, cls_id) in filtered_labels
                  if imageid_is_train.get(img_id, False)]
filtered_test = [(img_id, cls_id) for (img_id, cls_id) in filtered_labels
                 if not imageid_is_train.get(img_id, False)]
print("Filtered training images:", len(filtered_train))
print("Filtered test images:", len(filtered_test))

# Decide whether to include both train+test or only one split
# For now, we’ll include both in the subset. If you want only train or only test, replace
# `filtered_labels` below with `filtered_train` or `filtered_test` as needed.

# === STEP 5: Create subset folder + copy images ===
subset_images_dir = os.path.join(subset_root, "images")
os.makedirs(subset_images_dir, exist_ok=True)

# Create one subfolder per class
for cls_id, cls_name in id_to_name.items():
    os.makedirs(os.path.join(subset_images_dir, cls_name), exist_ok=True)

# Copy each filtered image
for (image_id, class_id) in filtered_labels:
    rel_path = imgid_to_path[image_id]
    orig_img_path = os.path.join(cub_root, "images", rel_path)
    dst_rel_folder = id_to_name[class_id]
    dst_folder = os.path.join(subset_images_dir, dst_rel_folder)
    dst_path = os.path.join(dst_folder, os.path.basename(rel_path))
    if not os.path.exists(orig_img_path):
        print(f"Warning: original image not found: {orig_img_path}")
        continue
    shutil.copy2(orig_img_path, dst_path)

# === STEP 6: Write reduced metadata files ===
# 6.1. images.txt
with open(os.path.join(subset_root, "images.txt"), "w") as fout:
    for (image_id, _) in filtered_labels:
        fout.write(f"{image_id} {imgid_to_path[image_id]}\n")

# 6.2. image_class_labels.txt
with open(os.path.join(subset_root, "image_class_labels.txt"), "w") as fout:
    for (image_id, class_id) in filtered_labels:
        fout.write(f"{image_id} {class_id}\n")

# 6.3. train_test_split.txt (optional)
with open(os.path.join(subset_root, "train_test_split.txt"), "w") as fout:
    for (image_id, _) in filtered_labels:
        is_train = int(imageid_is_train.get(image_id, False))
        fout.write(f"{image_id} {is_train}\n")

# 6.4. classes.txt (only the 24 classes)
with open(os.path.join(subset_root, "classes.txt"), "w") as fout:
    for cls_name, cls_id in sorted(name_to_id.items(), key=lambda kv: kv[1]):
        fout.write(f"{cls_id} {cls_name}\n")

# === STEP 7: Filter bounding_boxes.txt (optional) ===
orig_bbox_file = os.path.join(cub_root, "bounding_boxes.txt")
subset_bbox_file = os.path.join(subset_root, "bounding_boxes.txt")
with open(orig_bbox_file, "r") as fin, open(subset_bbox_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        image_id = int(line.split()[0])
        if image_id in filtered_image_ids:
            fout.write(line + "\n")

# === STEP 8: Filter part annotations ===
# 8.1. Copy parts/parts.txt (lookup table) unchanged
os.makedirs(os.path.join(subset_root, "parts"), exist_ok=True)
shutil.copy2(
    os.path.join(cub_root, "parts", "parts.txt"),
    os.path.join(subset_root, "parts", "parts.txt")
)

# 8.2. Filter parts/part_locs.txt
orig_part_locs = os.path.join(cub_root, "parts", "part_locs.txt")
subset_part_locs = os.path.join(subset_root, "parts", "part_locs.txt")
with open(orig_part_locs, "r") as fin, open(subset_part_locs, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if not parts:
            continue
        image_id = int(parts[0])
        if image_id in filtered_image_ids:
            fout.write(line)

# 8.3. Filter parts/part_click_locs.txt
orig_click_locs = os.path.join(cub_root, "parts", "part_click_locs.txt")
subset_click_locs = os.path.join(subset_root, "parts", "part_click_locs.txt")
with open(orig_click_locs, "r") as fin, open(subset_click_locs, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if not parts:
            continue
        image_id = int(parts[0])
        if image_id in filtered_image_ids:
            fout.write(line)

# === STEP 9: Filter attribute annotations ===
# 9.1. Copy attributes/attributes.txt and attributes/certainties.txt unchanged
os.makedirs(os.path.join(subset_root, "attributes"), exist_ok=True)
shutil.copy2(
    os.path.join(cub_root, "attributes", "attributes.txt"),
    os.path.join(subset_root, "attributes", "attributes.txt")
)
shutil.copy2(
    os.path.join(cub_root, "attributes", "certainties.txt"),
    os.path.join(subset_root, "attributes", "certainties.txt")
)

# 9.2. Filter attributes/image_attribute_labels.txt
orig_iattr_file = os.path.join(cub_root, "attributes", "image_attribute_labels.txt")
subset_iattr_file = os.path.join(subset_root, "attributes", "image_attribute_labels.txt")
with open(orig_iattr_file, "r") as fin, open(subset_iattr_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if not parts:
            continue
        image_id = int(parts[0])
        if image_id in filtered_image_ids:
            fout.write(line)

# 9.3. Filter attributes/class_attribute_labels_continuous.txt
orig_cattr_file = os.path.join(cub_root, "attributes", "class_attribute_labels_continuous.txt")
subset_cattr_file = os.path.join(subset_root, "attributes", "class_attribute_labels_continuous.txt")

# Read all 200 lines from the original file
with open(orig_cattr_file, "r") as f:
    all_cont_lines = f.read().splitlines()

# Build a list of class names in order (line index 0 → classes_list[0], etc.)
classes_list = [line.strip().split()[1] for line in open(classes_file, "r")]

# Map class_name → (0-based index into class_attribute_labels_continuous.txt)
name_to_index = {name: idx for idx, name in enumerate(classes_list)}

with open(subset_cattr_file, "w") as fout:
    # Write one line per target class, in ascending class_id order
    for cls_name, cls_id in sorted(name_to_id.items(), key=lambda kv: kv[1]):
        idx = name_to_index[cls_name]
        fout.write(all_cont_lines[idx] + "\n")

# === STEP 10: Optional CSV with relative paths and class IDs ===
rows = []
for (image_id, class_id) in filtered_labels:
    rel_path = imgid_to_path[image_id]  # e.g. "048.European_Goldfinch/…jpg"
    rows.append({
        "relative_path": rel_path,
        "class_id": class_id
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(subset_root, "subset_labels.csv")
df.to_csv(csv_path, index=False)
print(f"CSV with {len(df)} entries written to {csv_path}")

print("Subset creation complete. All images and metadata are under:", subset_root)
