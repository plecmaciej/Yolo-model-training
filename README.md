# YOLO Model Training

This project — **YOLO Model Training** — provides a complete set of Python scripts for **training, prediction, and visualization** of YOLO models (mainly YOLOv8, but it can be easily adapted to other YOLO versions).  

It was built using:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://github.com/pytorch/pytorch)

---

## Project Overview

The goal of this project is to **train and evaluate YOLO models** for object detection tasks.  
The included scripts allow you to:
- Train a YOLO model on a custom dataset  
- Run inference (prediction) on new images  
- Visualize detected classes  
- Evaluate detection accuracy  
- Filter or preprocess datasets  

A **trained model** is included in this repository:

```bash
runs/detect/care_symbols_v4/weights/best.pt
```

This model can detect and classify **laundry care symbols** (symbols found on clothing labels).  
You can use it directly for inference or continue training on your own data.

---

## Dataset

The dataset used to train this model was obtained from Roboflow:
[Care Symbols Dataset – Roboflow Universe](https://universe.roboflow.com/carescan/carelabelsfind/dataset/12)

If you wish to use your own dataset:
1. Visit [Roboflow Universe](https://universe.roboflow.com/)
2. Download any dataset in a YOLO-compatible format (e.g., YOLOv8 or YOLOv5)
3. Place it in the `datasets/` directory of this project

---

## Additional Resources

This project also includes a folder:
```bash
washing_symbols/
``` 
It contains **visual representations of laundry care symbols** used for easier visualization of detection results.

---

## Installation & Requirements

Before running any scripts, install the required dependencies:

```bash
pip install -r requirements.txt
```
If you wish to train or run inference on GPU (CUDA), make sure you have:
- A CUDA-capable GPU  
- Proper NVIDIA drivers   
- Installed CUDA Toolkit and cuDNN compatible with your PyTorch version  

If you don’t have CUDA support, simply run the model on CPU.
To do this, open your training script (e.g., train_model.py) and change:
```bash
device="cuda" to device="cpu"
```
---

## File Descriptions

### `train_model.py`
This is the **core training script** of the project.  
It handles the YOLO model training process, including parameter selection such as:
- number of epochs  
- image size  
- batch size  
- learning rate  
- device type (CPU / CUDA)

The most important configuration for training is specifying the **correct path to the dataset YAML file** (which defines your training and validation data) and selecting the desired **YOLO model type** (e.g., `yolov8m.pt`).  
The internal training logic follows the standard [PyTorch](https://pytorch.org/) and [Ultralytics YOLO](https://docs.ultralytics.com/) pipeline.

---

### `predict.py`
This script performs **inference (prediction)** using your trained YOLO model.  
It loads the model (e.g., from `runs/detect/care_symbols_v4/weights/best.pt`) and predicts the detected classes on provided images.  
Each prediction prints out the **detected class names with their confidence scores** — usually above **95% accuracy** for this trained model.

---

### `filtering_datasets.py`
This utility script is used to **clean and filter your dataset**.  
During dataset labeling, it sometimes happens that an entire image is incorrectly marked as one big bounding box — which is wrong.  
This script automatically detects such cases, removes them, and copies only correctly labeled data into a filtered dataset directory.

---

### `set_checking.py`
<img width="795" height="827" alt="image" src="https://github.com/user-attachments/assets/74c669b9-fb17-4d18-9e03-f427be3f3c03" />

This script allows you to **visually verify your dataset annotations**.  
It draws bounding boxes on images based on YOLO label files, so you can easily check whether each class is properly marked and aligned with the object in the image.

---

### `visualize.py`
<img width="1870" height="405" alt="image" src="https://github.com/user-attachments/assets/2280bb47-46bd-4a63-afec-8005a0ee8166" />

After predictions are made, this script **visualizes which classes were detected** using graphical icons.  
It takes the prediction output (label files) and displays the corresponding **laundry symbols** along with the number of occurrences for each class.  
The visualization uses the images from the `washing_symbols/` folder.

---

### `predict_and_visualize.py`
<img width="1121" height="747" alt="image" src="https://github.com/user-attachments/assets/428eefc3-e159-4770-a5d7-f5ee6d647b4c" />

This combined script performs **real-time prediction with visualization**.  
It runs inference and immediately shows the predicted results on screen — allowing you to see live how the model detects and classifies the laundry care symbols.

---

## Acknowledgements

This project makes use of the following open-source tools and datasets:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection framework used for model training and inference  
- [PyTorch](https://github.com/pytorch/pytorch) — deep learning library powering YOLO  
- [Roboflow Universe Dataset](https://universe.roboflow.com/carescan/carelabelsfind/dataset/12) — dataset used for training and testing the model  
