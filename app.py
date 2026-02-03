import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

st.title("Potato Leaf Disease Classifier 🌿")

uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((256, 256))  # match training size
    img_array = np.array(img)  # normalize
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    st.markdown(f"### Prediction: `{predicted_class}`")
    st.markdown(f"### Confidence: `{confidence}%`")