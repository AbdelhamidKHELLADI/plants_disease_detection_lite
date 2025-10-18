---
title: "🌿 Plant Diseases Classifier"
emoji: "🌱"
colorFrom: "green"
colorTo: "blue"
sdk: "streamlit"
app_file: "streamlit_app.py"
---


# 🌿 Plant Diseases Classifier (MobileNetV3 + TFLite)

[![Hugging Face Space](https://img.shields.io/badge/🚀-HuggingFace_Space-blue.svg)](https://huggingface.co/spaces/khhamid/plants-diseases-detection)


An optimized **plant disease detection** web app built using **TensorFlow**, **MobileNetV3**, and **Streamlit**, with **quantized TFLite deployment** for efficient inference on the web, cloud, and edge devices.  
This project demonstrates real-world model optimization, containerized deployment, and automated CI/CD to **Hugging Face Spaces** and **Docker Hub**.

---

## 🚀 Features

- ✅ **Fine-tuned MobileNetV3-Small** for accurate plant disease recognition  
- ⚡ **TFLite quantization (INT8)** for ultra-fast inference  
- 🌐 **Streamlit-based web interface** for image upload and prediction  
- 📦 **Dockerized environment** for reproducible builds  
- 🤗 **CI/CD pipeline** that automatically pushes to:
  - Docker Hub  
  - Hugging Face Models & Spaces  

---

## 🧠 Model Overview

| Component | Description |
|------------|-------------|
| **Architecture** | MobileNetV3-Small (fine-tuned) |
| **Dataset** | Plant Diseases Dataset |
| **Classes** | 38 different plant species and diseases |
| **Optimization** | INT8 Quantization (TensorFlow Lite) |
| **Performance** | 98%+ accuracy, **90% size reduction** |

---

## 📂 Project Structure

```bash
.
├── models/
│ └── mobilenet_int8.tflite # Quantized model
├── src/
│ └── app.py # Streamlit web app
├── class_names.txt # Class labels
├── requirements.txt # Python dependencies
├── Dockerfile # Container setup
├── .github/workflows/hf_deploy.yml # CI/CD workflow
└── README.md
```
---

## 🧰 Local Setup

###  Clone the repository
```bash
git clone https://github.com/<your-username>/plants-diseases-lite.git
cd plants-diseases-lite
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run the app
```bash
streamlit run src/app.py --server.port=7860
```
Then open: http://localhost:7860

## Docker Deployment
### 1 Build the image
```bash
docker build -t plants-diseases-lite .
```
### 2 Run the container
```bash
docker run -p 7860:7860 plants-diseases-lite
```
Open your browser at http://localhost:7860

## 🤗 Hugging Face Integration
This project includes a full CI/CD pipeline that automatically:

* Builds the app

* Pushes the Docker image to Docker Hub

* Uploads the quantized model to Hugging Face Model Hub

* Deploys the Streamlit app to Hugging Face Spaces

📈 Results
Metric | Keras Model | TFLite INT8 |
|-----|------------|-------------|
Accuracy |	98% |	98% |
Avg Inference Time |	- ms | ~2.7 ms |
Model Size |	22.4 MB	| **1.0 MB** |

## 🌍 Live Demo
Try it live on Hugging Face Spaces 👇
👉 [https://huggingface.co/spaces/khhamid/plants-diseases-lite-app](https://huggingface.co/spaces/khhamid/plants-diseases-detection)



## Future Work
* 📱 Deploy on edge & mobile devices using:

* ☁️ Add backend API for large-scale cloud predictions

* 🌾 Expand dataset with real-world agricultural images

* 📷 Enable live camera inference for instant in-field diagnosis

---
## 🪴 License
This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.

## Refs:
Check out the configuration reference at [https://huggingface.co/docs/hub/spaces-config-reference](https://huggingface.co/docs/hub/spaces-config-reference)

## ⭐ If you like this project, give it a star!
Your feedback helps improve open-source AI for agriculture 🌱

