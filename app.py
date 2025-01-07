import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
MODEL_PATH = 'C:/Users/sreevishak/Desktop/DUK/mini/models12/violence_detection_model.h5'
model = load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the target size (64x64x3)
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to make a prediction
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]  # Get the prediction score
    return prediction

# Streamlit app
st.title("Violence Detection from Images")
st.write("Upload an image to classify it as **Violence** or **Non-Violence**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
  
    # Predict the image
    prediction = predict_image(image)
    if prediction > 0.5:
        st.write("### Prediction: 🛑 Violence Detected")
        st.write(f"Confidence Score: {prediction:.2f}")
    else:
        st.write("### Prediction: ✅ Non-Violence")
        st.write(f"Confidence Score: {1 - prediction:.2f}")
