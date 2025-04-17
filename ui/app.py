import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Import preprocess functions for each model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Define your class names (update with your actual classes)
# Class labels
class_names = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumothorax"
]  # <-- update this

# Load model based on selection
@st.cache_resource
def load_selected_model(name):
    if name == 'EfficientNetB0':
        model_path = 'jobinajoy/xrayimagediseaseprediction/main/model/EfficientNetB0Model.h5'
        model = load_model(model_path, custom_objects={'preprocess_input': efficientnet_preprocess})
    elif name == "ResNet50":
        model_path = 'jobinajoy/xrayimagediseaseprediction/main/model/EfficientNetB0Model.h5'
        model = load_model(model_path, custom_objects={'preprocess_input': resnet_preprocess})
    else:
        model, accuracy = None, None
    return model, accuracy

# Streamlit UI
st.title("🩺 Abnormality Detection in Lungs Using XRay")

# Model selection
model_name = st.selectbox("Choose a model", ["EfficientNetB0", "ResNet50"])
model, model_accuracy = load_selected_model(model_name)

# Display model info
if model:
    st.markdown(f"**Loaded Model:** `{model_name}`")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Show result
        st.success(f"Predicted Abnormality: **{predicted_class}** ({confidence * 100:.2f}%)")
else:
    st.error("Model not loaded. Please select a valid model.")
