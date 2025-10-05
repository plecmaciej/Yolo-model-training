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

## Acknowledgements

This project makes use of the following open-source tools and datasets:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection framework used for model training and inference  
- [PyTorch](https://github.com/pytorch/pytorch) — deep learning library powering YOLO  
- [Roboflow Universe Dataset](https://universe.roboflow.com/carescan/carelabelsfind/dataset/12) — dataset used for training and testing the model  
