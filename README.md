# Cloud Cancer Tissue ViT Platform

A cloud-based deep learning platform for **cancer tissue image classification** using **PyTorch Vision Transformers (ViT)**. The project provides an end-to-end pipeline for training, evaluating, and deploying image classification models through a **FastAPI** service with optional **AWS S3** storage and **EC2** deployment.

---

## Features

* PyTorch Vision Transformer (ViT) image classifier
* Histopathology tissue image classification
* Training and validation pipeline
* Configurable training using YAML files
* FastAPI inference service
* Docker containerization
* AWS S3 integration for image storage
* Optional AWS EC2 deployment
* Probability visualization for all predicted classes
* Attention map visualization (planned)
* Modular and extensible project architecture

---

## Project Structure

```text
cloud-cancer-tissue-vit-platform/
│
├── configs/
│   ├── train.yaml
│   ├── inference.yaml
│   └── aws.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── docker/
│
├── models/
│   ├── checkpoints/
│   └── exported/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── predictions/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── aws/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── visualization/
│
├── tests/
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Workflow

```text
Histopathology Image
          │
          ▼
   Image Preprocessing
          │
          ▼
 Vision Transformer (ViT)
          │
          ▼
  Classification Probabilities
          │
          ▼
     FastAPI REST API
          │
          ▼
     AWS S3 (Optional)
          │
          ▼
     Web / Client Application
```

---

## Technology Stack

| Category         | Technologies             |
| ---------------- | ------------------------ |
| Language         | Python 3.12              |
| Deep Learning    | PyTorch, TorchVision     |
| Computer Vision  | OpenCV, Pillow           |
| Machine Learning | Vision Transformer (ViT) |
| API              | FastAPI, Uvicorn         |
| Cloud            | AWS S3, EC2              |
| Containerization | Docker                   |
| Configuration    | YAML                     |
| Testing          | PyTest                   |

---

## Planned Features

* Vision Transformer (ViT-B/16)
* Transfer Learning
* Data Augmentation
* Mixed Precision Training
* Early Stopping
* Checkpoint Management
* ONNX Export
* TorchScript Export
* Batch Inference
* Explainable AI (Attention Maps)
* Grad-CAM Visualization
* Docker Compose Deployment
* AWS Cloud Deployment

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/your_username/cloud-cancer-tissue-vit-platform.git
cd cloud-cancer-tissue-vit-platform
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
bash scripts/train.sh
```

Run the inference API:

```bash
bash scripts/run_api.sh
```

---

## Future Roadmap

* Support for Whole Slide Images (WSI)
* Multi-class cancer classification
* Swin Transformer implementation
* Model benchmarking
* Cloud-native deployment
* Interactive web interface
* Experiment tracking
* CI/CD pipeline

---

## License

This project is released under the MIT License.

---

## Author

William Popkov

Computer Vision • Machine Learning • Medical Imaging • Robotics
