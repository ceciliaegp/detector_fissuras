import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# --------------------------------------------------
# Carregar modelo
# --------------------------------------------------

@st.cache_resource
def carregar_modelo():
    return YOLO("best.pt")

model = carregar_modelo()

# --------------------------------------------------
# Função para desenhar o label real
# --------------------------------------------------

def desenhar_label_real(imagem, arquivo_label):
    img = imagem.convert("RGB")
    largura, altura = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    conteudo = arquivo_label.read().decode("utf-8")
    linhas = conteudo.splitlines()

    for linha in linhas:
        valores = linha.strip().split()

        if len(valores) < 3:
            continue

        coordenadas = list(map(float, valores[1:]))

        pontos = []

        for i in range(0, len(coordenadas), 2):
            x = coordenadas[i] * largura
            y = coordenadas[i + 1] * altura
            pontos.append((x, y))

        draw.polygon(
            pontos,
            fill=(0, 255, 0, 80),
            outline=(0, 255, 0, 255)
        )

    return Image.alpha_composite(
        img.convert("RGBA"),
        overlay
    )

# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("Detector de Fissuras")

uploaded_image = st.file_uploader(
    "Envie a imagem",
    type=["jpg", "jpeg", "png"]
)

uploaded_label = st.file_uploader(
    "Envie o label correspondente (.txt)",
    type=["txt"]
)

if uploaded_image is not None and uploaded_label is not None:

    imagem = Image.open(uploaded_image)

    # -------------------------------
    # Label real
    # -------------------------------

    img_real = desenhar_label_real(
        imagem,
        uploaded_label
    )

    # -------------------------------
    # Predição YOLO
    # -------------------------------

    results = model.predict(
        source=np.array(imagem),
        conf=0.25
    )

    imagem_resultado = results[0].plot()

    # -------------------------------
    # Exibição
    # -------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Imagem Original")
        st.image(imagem)

    with col2:
        st.subheader("Label Real")
        st.image(img_real)

    with col3:
        st.subheader("Predição YOLO")
        st.image(imagem_resultado)

    st.success("Processamento concluído.")
