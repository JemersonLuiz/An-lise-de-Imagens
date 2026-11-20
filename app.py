import logging
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from classificador import LIMIAR, carregar_rede, classificar, ler_foto

st.set_page_config(page_title="Triagem de materiais", layout="centered")
rede_em_cache = st.cache_resource(show_spinner=False)(carregar_rede)
AMOSTRAS = {
    "Lata de conserva": "metal.jpg",
    "Jornal dobrado": "papel.jpg",
    "Garrafa âmbar": "vidro.jpg",
}
PASTA = Path(__file__).parent / "amostras"


def selecionar_amostra(nome=None):
    st.session_state.amostra = nome

st.title("Triagem de materiais")
st.write("Fotografe um único resíduo para identificar papel, plástico, metal, vidro ou material orgânico.")

foto = st.file_uploader(
    "Selecionar fotografia",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    max_upload_size=12,
    on_change=selecionar_amostra,
    help="Até 12 MB e 24 megapixels. Enquadre o objeto inteiro e use boa iluminação.",
)
amostra = st.session_state.get("amostra")
if foto is None and amostra is None:
    st.divider()
    st.subheader("Testar com uma amostra")
    for coluna, (nome, arquivo) in zip(st.columns(3), AMOSTRAS.items()):
        with coluna:
            st.image(str(PASTA / arquivo), width="stretch", alt=nome)
            st.button(nome, on_click=selecionar_amostra, args=(nome,), width="stretch")
    st.stop()

if foto is None:
    foto = BytesIO((PASTA / AMOSTRAS[amostra]).read_bytes())
    st.button("Trocar amostra", on_click=selecionar_amostra)

try:
    imagem = ler_foto(foto)
except (OSError, Image.DecompressionBombError, SyntaxError):
    st.error("Não foi possível abrir a foto. Escolha uma imagem JPEG, PNG, WebP ou BMP válida.")
    st.stop()
except ValueError as erro:
    st.error(str(erro))
    st.stop()

st.divider()
previa, leitura = st.columns([1.1, 1], gap="large")
with previa:
    miniatura = imagem.copy()
    miniatura.thumbnail((800, 800))
    st.image(miniatura, width="stretch", alt="Fotografia enviada para classificação")

with leitura:
    st.caption("RESULTADO DA TRIAGEM")
    try:
        with st.spinner("Analisando a fotografia…"):
            material, escore = classificar(rede_em_cache(), imagem)
    except Exception:
        logging.exception("Falha ao carregar ou executar o classificador")
        st.error("Classificador indisponível. O primeiro uso precisa de internet para baixar o modelo.")
        if st.button("Tentar novamente"):
            st.rerun()
        st.stop()

    if escore < LIMIAR:
        st.subheader("Identificação inconclusiva")
        st.write("Tente outra foto com um único objeto, mais perto e em um fundo simples.")
    elif material is None:
        st.subheader("Fora das categorias")
        st.write("A rede associou a imagem a um resíduo fora dos cinco materiais da triagem.")
    else:
        st.subheader(material)
        st.write(f"Escore do modelo · {escore:.0%}")
