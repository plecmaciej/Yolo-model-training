from ultralytics import YOLO

tested_pictures_path = "datasets/care_symbols/test_filtered/fast"

model = YOLO("runs/detect/care_symbols_v4/weights/best.pt")
results = model.predict(
    source=tested_pictures_path,
    save=True,
    save_txt=True,
    conf=0.4
)

for i, result in enumerate(results):
    print(f"\n Picture {i+1}:")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f" - Class: {model.names[cls_id]}, %: {conf:.2f}")