# 🍇 Grape Leaf Disease Detection with Optimized Feature Extraction and Attention Mechanisms

## 📌 Project Overview

This project aims to detect and classify grape leaf diseases using advanced deep learning techniques. We have implemented an optimized CNN model enhanced with attention mechanisms to improve feature extraction and increase prediction accuracy. The system can help farmers and agricultural experts detect diseases early and reduce crop loss.

---

## 🧠 Key Highlights

- ✅ Image classification of grape leaf diseases
- ✅ Optimized feature extraction using CNN
- ✅ Use of attention mechanisms like CBAM/SE Block
- ✅ Achieves high accuracy and robustness
- ✅ Easily extendable and scalable model
- ✅ (Optional) Streamlit-based web app for real-time detection

---

## 🗂️ Dataset

The dataset includes grape leaf images labeled into four categories:

- Healthy
- Black Rot
- Esca (Black Measles)
- Leaf Blight

📦 **Dataset Source**: [PlantVillage Dataset (via Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)  
All images are resized to 224x224 pixels and augmented for better generalization.

---

## ⚙️ Tech Stack

| Category         | Tools Used                         |
|------------------|------------------------------------|
| Programming Lang | Python 3.x                         |
| Deep Learning    | TensorFlow / PyTorch               |
| Image Processing | OpenCV, Pillow (PIL)               |
| Attention Layers | CBAM, SE Block, Transformer Blocks |
| Visualization    | Matplotlib, Seaborn                |
| Deployment       | Streamlit / Flask (optional)       |

---

## 🏗️ Model Architecture

- Base Model: CNN (4 Convolutional blocks + Pooling)
- Dropout & Batch Normalization
- Attention Mechanism (CBAM / SE Block)
- Fully Connected Dense Layers
- Output Layer with Softmax Activation

> You can switch attention modules with simple plug-and-play.

---

## 🚀 Getting Started

### 🔧 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/grape-leaf-disease-detection.git
cd grape-leaf-disease-detection
pip install -r requirements.txt

🏋️‍♀️ Training the Model
python train.py --epochs 50 --batch_size 32 --attention cbam

🧪 Evaluate the Model
python test.py --model saved_model/model_cbam.h5

🌐 Run Web Application (Optional)
streamlit run app.py

📊 Results
| Model          | Accuracy  | Precision | Recall    |
| -------------- | --------- | --------- | --------- |
| CNN (Baseline) | 89.2%     | 88.5%     | 89.0%     |
| CNN + SE Block | 93.3%     | 92.7%     | 93.0%     |
| CNN + CBAM     | **94.6%** | **94.1%** | **94.3%** |

grape-leaf-disease-detection/
│
├── dataset/                   # Training and test images
├── models/                    # Saved models
├── attention_modules/         # Custom attention layers (CBAM, SE, etc.)
├── utils.py                   # Helper functions
├── train.py                   # Model training script
├── test.py                    # Model testing script
├── app.py                     # Web app (Streamlit)
├── requirements.txt           # Dependencies
└── README.md                  # This file

👨‍💻 Author
Sanjeet Kumar
Final Year B.Tech Project - 2025
📫 Email: sanjeetk0386@gmail.com
🔗 GitHub: github.com/meetwithsanju/
