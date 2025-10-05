from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    data_path = "datasets/care_symbols/data.yaml"

    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        device="cuda",
        batch=8,
        lr0=0.001,
        patience=20,
        name="care_symbols_v4"
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()