import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import time


st.title("🌿 MobileNet TFLite Image Classifier")
st.write("Upload an image to test your quantized MobileNet model.")


def load_labels(path):
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return None

labels = load_labels("class_names.txt")

def preprocess(image_array: np.ndarray) -> np.ndarray:
    """Replicate keras.applications.mobilenet_v3.preprocess_input"""
    image_array = image_array.astype(np.float32)
    image_array = image_array / 127.5 - 1.0  # scale to [-1, 1]
    return image_array

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
    st.image(image, caption="Uploaded Image", width="stretch")

    img = image.resize((224, 224))
    img = np.array(img)
    input_data = preprocess(img)
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):

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