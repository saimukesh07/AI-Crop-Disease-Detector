import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

MODEL_PATH = "crop_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Loading AI model... please wait"):
        gdown.download(
            "https://drive.google.com/uc?id=1kkO9yWWey8POuSpvvWhppZ2EYCL7t-Ho",
            MODEL_PATH,
            quiet=False
        )

model = tf.keras.models.load_model(MODEL_PATH)
with open("class_names.json") as f:
    class_names = json.load(f)

treatments = {
    "Tomato_Late_blight": "Apply copper-based fungicide. Remove infected leaves immediately.",
    "Tomato_Early_blight": "Use chlorothalonil fungicide. Ensure proper spacing for airflow.",
    "Corn_Common_rust": "Apply fungicide early. Use rust-resistant seed varieties.",
    "Potato_Late_blight": "Remove infected plants. Apply mancozeb fungicide.",
    "Pepper__bell___Bacterial_spot": "Use copper spray. Avoid overhead watering.",
}

st.set_page_config(page_title="🌿 Crop Disease Detector", layout="centered")
st.title("🌿 AI Crop Disease Detector")
st.write("Upload a leaf photo to instantly detect disease and get treatment advice!")

uploaded_file = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing leaf..."):
        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = class_names[np.argmax(predictions)]

    display_name = predicted_class.replace("_", " ").replace("  ", " → ")

    st.success(f"🔬 Detected: **{display_name}**")
    st.info(f"📊 Confidence: **{confidence:.1f}%**")

    st.subheader("💊 Recommended Treatment")
    treatment = treatments.get(predicted_class,
                "Consult a local agricultural expert for treatment advice.")
    st.warning(treatment)

    if "healthy" in predicted_class.lower():
        st.balloons()
        st.success("🎉 Great news! Your crop appears healthy!")
