import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

# Download model from Google Drive if not present
MODEL_PATH = "crop_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Loading AI model... please wait"):
        # Replace with your Google Drive file ID
        gdown.download(
            "https://drive.google.com/uc?id=YOUR_FILE_ID",
            MODEL_PATH,
            quiet=False
        )

model = tf.keras.models.load_model(MODEL_PATH)
with open("class_names.json") as f:
    class_names = json.load(f)