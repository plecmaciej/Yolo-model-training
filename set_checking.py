import os
import cv2
from ultralytics.utils.plotting import Annotator

img_dir = "datasets/care_symbols/train/images/"
label_dir = "datasets/care_symbols/train/labels/"

for img_name in os.listdir(img_dir):
    if img_name.endswith((".jpg", ".png")):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        if os.path.exists(label_path):
            annotator = Annotator(image)
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    annotator.box_label([x1, y1, x2, y2], str(int(class_id)))

            cv2.imshow("Image with labels", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()