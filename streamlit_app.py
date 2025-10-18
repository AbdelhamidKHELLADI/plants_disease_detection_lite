import os
import time
import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

os.environ["STREAMLIT_CACHE_DIR"] = "/tmp/streamlit-cache"
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/streamlit-config"
os.makedirs("/tmp/streamlit-cache", exist_ok=True)
os.makedirs("/tmp/streamlit-config", exist_ok=True)


st.set_page_config(page_title="🌿 MobileNet TFLite Classifier", layout="centered")
st.title("🌿 MobileNet TFLite Image Classifier")
st.write("Upload an image to test your quantized MobileNet model.")

def load_labels(path):
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return None

labels = load_labels("class_names.txt")

# ==============================
# Preprocess image for MobileNetV3
# ==============================
def preprocess(image_array: np.ndarray) -> np.ndarray:
    image_array = image_array.astype(np.float32)
    image_array = image_array / 127.5 - 1.0  # scale to [-1, 1]
    return image_array

# ==============================
# Load TFLite model with caching
# ==============================
@st.cache_resource
def load_tflite_model():
    interpreter = Interpreter(model_path="models/mobilenet_int8.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):

            img = image.resize((224, 224))
            img_array = np.array(img)
            input_data = preprocess(img_array)
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

       
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            inference_time = (time.time() - start) * 1000

            top_k = preds.argsort()[-3:][::-1]
            st.markdown("### 🌱 Predictions:")
            for i in top_k:
                label = labels[i] if labels else f"Class {i}"
                st.write(f"**{label}** — {preds[i] * 100:.2f}%")
                if preds[i]==1:
                    break
            st.info(f"⚡ Inference Time: {inference_time:.2f} ms")
