import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import pickle
from utils import clean_image, get_prediction, make_results

# Load the Trained Model
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# Removing Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Check if the model is already trained and saved
model_path = 'plant-disease-model.pkl'
model_exists = st.sidebar.checkbox('Load Pretrained Model', value=False)

# Load the Model
if model_exists:
    model = load_model(model_path)
else:
    model = None

# Title and Description
st.title('Plant Disease Detection')
st.write("Just upload an image of your plant's leaf.")

# Setting the files that can be uploaded
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])

# If there is an uploaded file, start making predictions
if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Detecting Image")
    my_bar = st.progress(0)
    i = 0

    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(Image.fromarray(
        np.array(image)).resize((700, 400), Image.ANTIALIAS)), width=None)
    my_bar.progress(i + 40)

    # Cleaning the image
    image = clean_image(image)

    # Making the predictions if model is loaded
    if model is not None:
        predictions, predictions_arr = get_prediction(model, image)
        my_bar.progress(i + 30)

        # Making the results
        result = make_results(predictions, predictions_arr)

        # Removing progress bar and text after prediction is done
        my_bar.progress(i + 30)
        progress.empty()
        i = 0

        # Show the results
        st.write(f"The plant {result['status']} with {result['prediction']} prediction.")
