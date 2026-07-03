import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

# ==========================
# PATH FIX — THE CORE BUG
# ==========================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "crop_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

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

/* ── Force ALL text on green background to be dark & readable ── */
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp div,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #1a1a1a !important;
}

/* Info / tip box text */
.stAlert p,
.stAlert span,
.stAlert div {
    color: #1a1a1a !important;
}

/* Bullet point text */
.stMarkdown li {
    color: #1a1a1a !important;
    font-weight: 500;
}

/* Caption text */
.stApp [data-testid="stCaptionContainer"] p {
    color: #1a1a1a !important;
}

/* Main Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1b5e20 !important;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #1b5e20 !important;
    margin-bottom: 30px;
    font-weight: 600;
}

/* Result Card */
.card {
    background: #e0e0e0;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-top: 20px;
}

.card h2, .card h3, .card p, .card b {
    color: #1a1a1a !important;
}

/* Upload Section */
[data-testid="stFileUploader"] {
    background: #e0e0e0;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #1a1a1a !important;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #d6d6d6;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #1a1a1a !important;
}

/* Button */
.stButton>button {
    background-color: #2e7d32;
    color: #ffffff !important;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}

/* Healthy card */
.card-healthy {
    background: #e8f5e9;
    border-left: 6px solid #43a047;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.10);
    margin-top: 20px;
}

.card-healthy h2, .card-healthy p, .card-healthy b {
    color: #1a1a1a !important;
}

/* Disease card */
.card-disease {
    background: #fff8e1;
    border-left: 6px solid #f9a825;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.10);
    margin-top: 20px;
}

.card-disease h2, .card-disease p, .card-disease b {
    color: #1a1a1a !important;
}

/* Low confidence warning */
.card-warning {
    background: #fce4ec;
    border-left: 6px solid #e53935;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.10);
    margin-top: 20px;
}

.card-warning h2, .card-warning p, .card-warning b {
    color: #1a1a1a !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL (cached)
# ==========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading AI model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1kkO9yWWey8POuSpvvWhppZ2EYCL7t-Ho",
                MODEL_PATH,
                quiet=False
            )
    return tf.keras.models.load_model(MODEL_PATH)

# ==========================
# LOAD CLASS NAMES (cached)
# ==========================
@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(
            f"❌ `class_names.json` not found.\n\n"
            f"Expected location: `{CLASS_NAMES_PATH}`\n\n"
            "Make sure it is in the **same folder** as `streamlit_app.py`."
        )
        st.stop()
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)

model       = load_model()
class_names = load_class_names()

# ==========================
# TREATMENT DATABASE
# ==========================
treatments = {
    "Tomato_Late_blight":               "Apply copper-based fungicide. Remove infected leaves immediately.",
    "Tomato_Early_blight":              "Use chlorothalonil fungicide. Ensure proper spacing for airflow.",
    "Tomato_Bacterial_spot":            "Apply copper-based bactericide. Avoid overhead irrigation.",
    "Tomato_Leaf_Mold":                 "Improve ventilation. Apply mancozeb or copper fungicide.",
    "Tomato_Septoria_leaf_spot":        "Remove infected leaves. Apply fungicide containing mancozeb.",
    "Tomato_Spider_mites":              "Use miticide spray. Increase humidity around plants.",
    "Tomato_Target_Spot":               "Apply fungicide. Remove heavily infected leaves.",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants. Control whitefly population.",
    "Tomato_Tomato_mosaic_virus":       "Remove infected plants. Disinfect tools regularly.",
    "Corn_Common_rust":                 "Apply fungicide early. Use rust-resistant seed varieties.",
    "Corn_Gray_leaf_spot":              "Apply strobilurin fungicide. Rotate crops annually.",
    "Corn_Northern_Leaf_Blight":        "Apply fungicide at early tassel stage. Use resistant hybrids.",
    "Potato_Late_blight":               "Remove infected plants. Apply mancozeb fungicide.",
    "Potato_Early_blight":              "Apply chlorothalonil. Ensure good drainage and air circulation.",
    "Pepper__bell___Bacterial_spot":    "Use copper spray. Avoid overhead watering.",
    "Apple_Apple_scab":                 "Apply fungicide during wet weather. Remove fallen leaves.",
    "Apple_Black_rot":                  "Prune infected branches. Apply captan fungicide.",
    "Apple_Cedar_apple_rust":           "Remove nearby cedar trees if possible. Apply fungicide.",
    "Grape_Black_rot":                  "Remove mummified fruit. Apply mancozeb fungicide.",
    "Grape_Esca_(Black_Measles)":       "Prune infected wood. No effective chemical cure; manage vineyard hygiene.",
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper-based fungicide. Improve air circulation.",
}

DEFAULT_TREATMENT = "Consult a local agricultural expert for proper diagnosis and treatment."

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("🌱 About This App")

st.sidebar.info("""
This AI system detects crop diseases from leaf images.

### Supported Crops
🍅 Tomato  🥔 Potato  
🌽 Corn    🫑 Pepper  
🍎 Apple   🍇 Grape  

### Built With
• TensorFlow  
• Streamlit  
• Deep Learning CNN  
""")

st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider(
    "⚙️ Confidence threshold",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Predictions below this % are flagged as uncertain."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Classes loaded:** {len(class_names)}")
st.sidebar.markdown(f"**Model:** `crop_disease_model.h5`")

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
    type=["jpg", "jpeg", "png", "webp"]
)

# ==========================
# PREDICTION
# ==========================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Leaf", width=350)

    img        = image.resize((224, 224))
    img_array  = np.array(img, dtype=np.float32) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analysing leaf image..."):
        predictions     = model.predict(img_array, verbose=0)
        confidence      = float(np.max(predictions)) * 100
        predicted_class = class_names[int(np.argmax(predictions))]

    display_name = predicted_class.replace("_", " ")
    is_healthy   = "healthy" in predicted_class.lower()
    is_uncertain = confidence < confidence_threshold

    if is_uncertain:
        st.markdown(f"""
        <div class="card-warning">
            <h2>⚠️ Low Confidence Prediction</h2>
            <p><b>Best guess:</b> {display_name}</p>
            <p><b>Confidence:</b> {confidence:.2f}% — below your {confidence_threshold}% threshold</p>
            <p>Try a clearer, better-lit photo of just the leaf.</p>
        </div>
        """, unsafe_allow_html=True)

    elif is_healthy:
        st.markdown(f"""
        <div class="card-healthy">
            <h2>✅ Crop Appears Healthy</h2>
            <p><b>Detected:</b> {display_name}</p>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
            <p>No disease signs detected. Continue regular care and monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    else:
        treatment = treatments.get(predicted_class, DEFAULT_TREATMENT)
        st.markdown(f"""
        <div class="card-disease">
            <h2>🔬 Disease Detected</h2>
            <p><b>Disease:</b> {display_name}</p>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <h3>💊 Recommended Treatment</h3>
            <p>{treatment}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Confidence")
    st.progress(int(confidence))
    st.caption(f"{confidence:.2f}% confident")

    with st.expander("🔎 See top-5 predictions"):
        top5_idx   = np.argsort(predictions[0])[::-1][:5]
        top5_names = [class_names[i].replace("_", " ") for i in top5_idx]
        top5_probs = [float(predictions[0][i]) * 100 for i in top5_idx]

        for name, prob in zip(top5_names, top5_probs):
            st.markdown(f"**{name}**")
            st.progress(int(prob))
            st.caption(f"{prob:.2f}%")

# ==========================
# PLACEHOLDER WHEN NO IMAGE
# ==========================
else:
    st.info("👆 Upload a leaf image above to get started.")
    st.markdown("""
    **Tips for best results:**
    - 📷 Use a clear, well-lit photo
    - 🍃 Focus tightly on the leaf — avoid busy backgrounds
    - 📐 Any orientation is fine — the model handles rotation
    - 🖼️ Supported formats: JPG, PNG, WEBP
    """)

# ==========================
# FOOTER
# ==========================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#1b5e20; font-weight:600;'>
Built using Deep Learning 🌱
</div>
""", unsafe_allow_html=True)
