import streamlit as st 
from fastai.vision.all import *

st.title("Transportni klassifikasiya qiluvchi model")

# st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg', 'gif'])

import pathlib 
import plotly.express as px
import platform

plt = platform.system()
if plt=='Linux':pathlib.WindowsPath=pathlib.PosixPath
st.title("Transport classification")

file = st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg', 'gif'])

if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('transports_model.pkl')
    prediction, pred_idx, probs = model.predict(img)
    st.success(f'Bashorat: {prediction}')