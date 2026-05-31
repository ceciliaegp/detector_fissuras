import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --------------------------------------------------
# Configuração da página
# --------------------------------------------------

st.set_page_config(
    page_title="Reconhecimento de Fissuras",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# Carregar modelo
# --------------------------------------------------

@st.cache_resource
def carregar_modelo():
    return YOLO("best.pt")

model = carregar_modelo()

# --------------------------------------------------
# Cabeçalho
# --------------------------------------------------

st.title("Reconhecimento de Fissuras com YOLOv8")
st.markdown("**Autora:** Cecilia Giuffra")
st.markdown(
    "Aplicação desenvolvida para detecção e segmentação de fissuras em imagens utilizando YOLOv8."
)
st.divider()

# --------------------------------------------------
# Upload da imagem
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Predição
# --------------------------------------------------

if uploaded_file is not None:

    imagem = Image.open(uploaded_file)

    st.subheader("Imagem enviada")
    st.image(imagem)

    with st.spinner("Analisando imagem..."):

        results = model.predict(
            source=np.array(imagem),
            conf=0.25
        )

    imagem_resultado = results[0].plot()

    st.subheader("Fissuras detectadas")
    st.image(imagem_resultado)

    st.success("Processamento concluído.")
```
