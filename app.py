import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

import pickle
#filename='proj_main_code-Copy1.ipynb'
#pickle.dump(model,open(filename,'wb'))
# Load the trained model
model_path = 'trained_model'  # Replace with the actual path to your model
model = load_model(model_path)

# Function to get class names
def names(number):
    if number == 0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'

# Streamlit app
st.title("Brain Tumor Classification App")

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose a .jpg image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))
    x = np.array(img)
    x = x.reshape(1, 128, 128, 3) / 255.0  # Normalize to [0, 1]

    # Make predictions using the model
    res = model.predict_on_batch(x)
    classification = np.argmax(res)

    # Get the human-readable label
    result_label = names(classification)

    # Display the image and result
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.success(f"Prediction: {result_label} with {res[0][classification] * 100:.2f}% confidence.")
