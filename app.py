import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

st.header('Image Classification Model')

# Load the pre-trained model
model = load_model('E:/ImageClassification/Image_classify_new2.keras')

# List of categories
data_cat = ['Group12', 'Sadabahar', 'Talmon', 'Xmas', 'aloevera', 'bamboogGrass', 'banana', 'climbers', 'giant', 'grass', 'group_round', 'simple', 'small_group']

# Dimensions for resizing the uploaded image
img_height = 180
img_width = 180

# Function to classify the uploaded image
def classify_image(image_path):
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = np.expand_dims(img_arr, axis=0)
    
    # Predict the class probabilities
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    
    return data_cat[np.argmax(score)], np.max(score) * 100

# File uploader for image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Display and classify the uploaded image
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=200)
    
    # Save the uploaded image to a temporary location
    with open(os.path.join("temp_images", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Get the path of the saved image
    image_path = os.path.join("temp_images", uploaded_file.name)
    
    # Classify the uploaded image
    predicted_class, accuracy = classify_image(image_path)
    
    # Display the prediction result
    st.write('The plant in the image is ' + predicted_class)
    st.write('With an accuracy of ' + str(accuracy))
