import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import gdown
import os

MODEL_PATH = "crop_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Loading AI model...please wait"):
        gdown.download("https://drive.google.com/uc?id=1kkO9yWWey8POuSpvvWhppZ2EYCL7t-Ho", MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)
with open("class_names.json") as f:
    class_names = json.load(f)
