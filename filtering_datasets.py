import os
import shutil

labels_dir = "datasets/care_symbols/test/labels"
images_dir = "datasets/care_symbols/test/images"
filtered_labels_dir = "datasets/care_symbols/test_filtered/labels"
filtered_images_dir = "datasets/care_symbols/test_filtered/images"

os.makedirs(filtered_labels_dir, exist_ok=True)
os.makedirs(filtered_images_dir, exist_ok=True)

# this script chcecks if the labels are correct for datasets
def is_full_image_box(line):
    # striping the values from line in format <class_id> <x_center> <y_center> <width> <height>
    parts = line.strip().split()
    if len(parts) != 5:
        return False

    _, x, y, w, h = map(float, parts)
    if abs(x - 0.5) < 0.01 and abs(y - 0.5) < 0.01 and abs(w - 1) < 0.01 and abs(h - 1) < 0.01:
        return True
    return False


for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, "r") as f:
        lines = f.readlines()

    if any(is_full_image_box(line) for line in lines):
        print(f"Dropping {label_file}")
        continue

    shutil.copy(label_path, os.path.join(filtered_labels_dir, label_file))
    image_file = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(images_dir, image_file)

    if os.path.exists(image_path):
        shutil.copy(image_path, os.path.join(filtered_images_dir, image_file))
    else:
        print(f"There is no picture for {label_file}")

print("Filtering is over")