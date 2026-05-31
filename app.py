import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def carregar_modelo():
  return YOLO("best.pt")
  
model = carregar_modelo()
st.title("Detector de Fissuras")

uploaded_file = st.file_uploader("Envie uma imagem",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  imagem = Image.open(uploaded_file)

  st.subheader("Imagem enviada")
  st.image(imagem)
  
  results = model.predict(source=np.array(imagem),conf=0.25)

  imagem_resultado = results[0].plot()
  
  st.subheader("Resultado")
  st.image(imagem_resultado)
