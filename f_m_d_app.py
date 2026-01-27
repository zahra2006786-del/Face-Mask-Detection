import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title='Face Mask Detection', layout='centered')
st.title('😷 Face Mask Detection Web App Using CNN 🧠')

# Loading model 
def load_model():
    model_path = 'tf_model.keras'
    if not os.path.exists(model_path):
        st.error('Model not found')
        st.stop()
        
    model = tf.keras.models.load_model('tf_model.keras', compile=False)
    return model

model = load_model()

image = st.file_uploader('Upload an Image of face with mask or without mask',
                         type = ['Jpeg', 'jpg', 'png', 'bmp', 'tiff', 'tif', 'webp'])

if image is not None:
    image = Image.open(image).convert('RGB')
    st.image(image, caption='Your Image', width= 500)


    image = image.resize((128, 128))
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = img / 255.0
    processed_img = tf.expand_dims(img, axis = 0)
    predict = model.predict(processed_img)[0]
    label = "❌ No Mask" if predict >= 0.5 else "😷 With Mask"

    st.subheader('Prediction Result')
    st.success(label)