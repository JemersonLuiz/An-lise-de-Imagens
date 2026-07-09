from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.utils import CLASSES_PATH, MODEL_PATH, resolve_project_path


def model_is_ready() -> bool:
    return resolve_project_path(MODEL_PATH).exists() and resolve_project_path(CLASSES_PATH).exists()


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        return Path(temporary_file.name)


st.set_page_config(page_title="Classificador de Residuos com IA", layout="centered")
st.title("Classificador de Res\u00edduos com IA")
st.write(
    "Envie uma imagem de um residuo reciclavel para receber a classe prevista "
    "e a confianca do modelo treinado."
)

if not model_is_ready():
    st.warning("Modelo ainda nao treinado. Execute `python src/train.py` antes de usar a interface.")

uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem enviada", use_container_width=True)

    if model_is_ready():
        temporary_path = save_uploaded_file(uploaded_file)
        try:
            from src.predict import predict_image

            with st.spinner("Processando imagem..."):
                result = predict_image(temporary_path)

            st.subheader("Resultado")
            st.write(f"Classe prevista: **{result['classe_prevista']}**")
            st.write(f"Confianca: **{result['confianca']:.2f}%**")

            st.subheader("Probabilidades")
            for class_name, probability in result["probabilidades"].items():
                st.write(f"{class_name}: {probability:.2f}%")
                st.progress(min(max(probability / 100, 0), 1))
        except Exception as exc:
            st.error(f"Nao foi possivel classificar a imagem: {exc}")
        finally:
            temporary_path.unlink(missing_ok=True)
