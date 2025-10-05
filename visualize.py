import os
import cv2
import yaml
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

labels_dir = 'runs/detect/predict3/labels'       # files with predicted symbols
symbols_dir = 'washing_symbols'                 # file with washing symbols
yaml_path = 'datasets/care_symbols/data.yaml'   # file that contains the names of classes

#upload names of classes
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml.get('names', {})

#list of labels after prediction
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

for label_file in label_files:
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        lines = f.readlines()
        class_ids = [int(line.strip().split()[0]) for line in lines]

    if not class_ids:
        print(f"Brak wykryć w: {label_file}")
        continue

    counter = Counter(class_ids)

    images = []
    titles = []

    for cls_id, count in counter.items():
        class_name = class_names[cls_id] if isinstance(class_names, list) else class_names.get(str(cls_id), f'class_{cls_id}')
        symbol_path = os.path.join(symbols_dir, f"{class_name}.png")

        if not os.path.exists(symbol_path):
            print(f"There is no image for a class '{class_name}' ({cls_id})")
            continue

        img = cv2.imread(symbol_path, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 4:
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]
            white_background = 255 * np.ones_like(rgb_channels, dtype=np.uint8)
            mask = alpha_channel[:, :, np.newaxis] / 255.0
            img = (rgb_channels * mask + white_background * (1 - mask)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        titles.append(f"{class_name} ({count}×)")

    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    img_name = label_file.replace(".txt", "")
    plt.suptitle(f"Detected symbols in: {img_name}")
    plt.tight_layout()
    plt.show()
