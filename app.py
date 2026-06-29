import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #d4fc79, #96e6a1);
}

/* Main Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1b5e20;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #2e7d32;
    margin-bottom: 30px;
}

/* Result Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-top: 20px;
}

/* Upload Section */
[data-testid="stFileUploader"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Button */
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "crop_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Downloading AI model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1kkO9yWWey8POuSpvvWhppZ2EYCL7t-Ho",
            MODEL_PATH,
            quiet=False
        )

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# LOAD CLASS NAMES
# ==========================
with open("class_names.json") as f:
    class_names = json.load(f)

# ==========================
# TREATMENT DATABASE
# ==========================
treatments = {
    "Tomato_Late_blight": "Apply copper-based fungicide. Remove infected leaves immediately.",
    "Tomato_Early_blight": "Use chlorothalonil fungicide. Ensure proper spacing for airflow.",
    "Corn_Common_rust": "Apply fungicide early. Use rust-resistant seed varieties.",
    "Potato_Late_blight": "Remove infected plants. Apply mancozeb fungicide.",
    "Pepper__bell___Bacterial_spot": "Use copper spray. Avoid overhead watering.",
}

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("🌱 About This App")

st.sidebar.info("""
This AI system detects crop diseases from leaf images.

### Supported Crops

🍅 Tomato  
🥔 Potato  
🌽 Corn  
🫑 Pepper  

### Built With

• TensorFlow  
• Streamlit  
• Deep Learning CNN  
""")

# ==========================
# HEADER
# ==========================
st.markdown("""
<div class="title">
🌿 Crop Disease Detector AI
</div>

<div class="subtitle">
Upload a leaf image and detect plant disease instantly using Artificial Intelligence
</div>
""", unsafe_allow_html=True)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "📸 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# PREDICTION
# ==========================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # center image
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(image, caption="Uploaded Leaf", width=350)

    # preprocess
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    with st.spinner("🔍 Analyzing leaf image..."):

        predictions = model.predict(img_array)

        confidence = np.max(predictions) * 100

        predicted_class = class_names[np.argmax(predictions)]

    display_name = predicted_class.replace("_", " ")

    # RESULT CARD
    st.markdown(f"""
    <div class="card">

    <h2>🔬 Disease Detection Result</h2>

    <p><b>Detected Disease:</b> {display_name}</p>

    <p><b>Confidence Score:</b> {confidence:.2f}%</p>

    </div>
    """, unsafe_allow_html=True)

    # Progress Bar
    st.subheader("📊 Prediction Confidence")
    st.progress(int(confidence))

    # Treatment
    treatment = treatments.get(
        predicted_class,
        "Consult a local agricultural expert for proper treatment."
    )

    st.markdown(f"""
    <div class="card">

    <h3>💊 Recommended Treatment</h3>

    <p>{treatment}</p>

    </div>
    """, unsafe_allow_html=True)

    # Healthy detection
    if "healthy" in predicted_class.lower():

        st.balloons()

        st.success("🎉 Great news! The crop appears healthy.")

# ==========================
# FOOTER
# ==========================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#1b5e20'>

Built using Deep Learning 🌱

</div>
""", unsafe_allow_html=True)
