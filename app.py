
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.keras")

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = load_model()

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image to predict whether it is a cat or dog.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array, verbose=0)[0][0])

    # Dataset labels: 0 = cat, 1 = dog
    predicted_index = 1 if probability >= 0.5 else 0
    confidence = probability if predicted_index == 1 else 1 - probability

    st.subheader(f"Prediction: {class_names[predicted_index]}")
    st.metric("Model confidence", f"{confidence * 100:.2f}%")
