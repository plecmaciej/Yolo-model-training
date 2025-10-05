import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import yaml


model_path = 'runs/detect/care_symbols_v4/weights/best.pt'
test_folder = 'datasets/care_symbols/test_filtered/images'
symbols_folder = 'washing_symbols'
data_yaml_path = 'datasets/care_symbols/data.yaml'

print("Model istnieje:", os.path.exists(model_path))
print("Test folder zawiera pliki:", os.listdir(test_folder))
print("Washing symbols:", os.listdir(symbols_folder))

# load the names of classes
with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

model = YOLO(model_path)

for img_file in os.listdir(test_folder):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(test_folder, img_file)
    results = model(img_path)[0]

    # count classes
    class_counts = {}
    for box in results.boxes.data:
        cls_id = int(box[5])
        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    # prepare visualization
    fig = plt.figure(figsize=(10, 6))
    spec = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # oryginalne większe, symbole mniejsze

    # original photo
    ax0 = plt.subplot(spec[0])
    image = Image.open(img_path).convert("RGB")
    ax0.imshow(image)
    ax0.set_title(f'Oryginalne: {img_file}')
    ax0.axis('off')

    # detected symbols
    ax1 = plt.subplot(spec[1])
    ax1.axis('off')
    symbol_imgs = []
    labels = []

    for cls_id, count in class_counts.items():
        class_name = class_names[cls_id]
        symbol_path = os.path.join(symbols_folder, f"{class_name}.png")

        if os.path.exists(symbol_path):
            symbol_img = cv2.imread(symbol_path, cv2.IMREAD_UNCHANGED)

            # decreasing the symbols
            symbol_img = cv2.resize(symbol_img, (64, 64))

            # converting color
            if symbol_img.shape[2] == 4:  # PNG z alfa
                alpha = symbol_img[:, :, 3] / 255.0
                rgb = symbol_img[:, :, :3]
                background = np.ones_like(rgb, dtype=np.uint8) * 255
                blended = (rgb * alpha[:, :, None] + background * (1 - alpha[:, :, None])).astype(np.uint8)
            else:
                blended = symbol_img

            symbol_imgs.append(blended)
            labels.append(f"{class_name} (x{count})")

    # Stwórz grid symboli
    rows = len(symbol_imgs)
    result_img = np.ones((rows * 70, 80, 3), dtype=np.uint8) * 255

    for i, (sym_img, label) in enumerate(zip(symbol_imgs, labels)):
        y = i * 70
        result_img[y:y+64, 8:72] = sym_img
        cv2.putText(result_img, label, (2, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)

    ax1.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Detected symbols ')

    plt.tight_layout()
    plt.show()
