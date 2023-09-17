import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('./bgs/bg.png')

# set title
st.title('Klasifikacija upale pluća')

# set header
st.header('Molimo vas da postavite rendgensku sliku grudnog koša')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/pneumonia_classifier.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # Formatiranje rezultata sa CSS stilizacijom
    st.markdown(
        f"<div style='text-align: center; border: 2px solid #3498db; padding: 10px; border-radius: 5px;'>"
        f"<h2>{class_name}</h2>"
        f"<h3>Score: {int(conf_score * 1000) / 10}%</h3>"
        f"</div>",
        unsafe_allow_html=True
    )