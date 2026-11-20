import logging
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from classificador import LIMIAR, carregar_rede, classificar, ler_foto

st.set_page_config(page_title="Triagem de materiais", layout="wide")
rede_em_cache = st.cache_resource(show_spinner=False)(carregar_rede)
RAIZ = Path(__file__).parent
AMOSTRAS = {
    "Lata de conserva": "metal.jpg",
    "Jornal dobrado": "papel.jpg",
    "Garrafa âmbar": "vidro.jpg",
}


def selecionar_amostra(nome=None):
    st.session_state.amostra = nome


def exibir_abertura(inicio):
    destino, rotulo = ("amostras", "Amostras") if inicio else ("resultado", "Resultado")
    st.html(f"""
        <header class="barra-triagem" id="inicio"><nav class="barra-conteudo" aria-label="Navegação principal">
            <a class="marca" href="#inicio" target="_self">Triagem</a>
            <div class="barra-links">
                <a href="#analisar" target="_self">Analisar foto</a>
                <a href="#{destino}" target="_self">{rotulo}</a>
            </div>
        </nav></header>
    """)
    if inicio:
        st.html("""
            <section class="capa" aria-labelledby="titulo-capa"><div class="capa-conteudo">
                <h1 id="titulo-capa">Triagem de <em>materiais.</em></h1>
                <p>Identifique papel, plástico, metal, vidro ou resíduos orgânicos a partir de uma fotografia.</p>
                <a class="acao" href="#analisar" target="_self">Analisar uma foto <span aria-hidden="true">→</span></a>
            </div></section>
        """)


def exibir_galeria():
    with st.container(key="galeria"):
        st.divider()
        st.header("Testar com uma amostra", anchor="amostras")
        for coluna, (nome, arquivo) in zip(st.columns(3), AMOSTRAS.items()):
            with coluna:
                st.image(str(RAIZ / "amostras" / arquivo), width="stretch", alt=nome)
                st.button(nome, on_click=selecionar_amostra, args=(nome,), width="stretch")


def exibir_resultado(foto):
    st.divider()
    st.header("Resultado da triagem", anchor="resultado")
    try:
        imagem = ler_foto(foto)
    except (OSError, Image.DecompressionBombError, SyntaxError):
        st.error("Não foi possível abrir a foto. Escolha uma imagem JPEG, PNG, WebP ou BMP válida.")
        return
    except ValueError as erro:
        st.error(str(erro))
        return

    previa, leitura = st.columns([1.1, 1], gap="large")
    with previa:
        miniatura = imagem.copy()
        miniatura.thumbnail((800, 800))
        st.image(miniatura, width="stretch", alt="Fotografia enviada para classificação")
    with leitura:
        try:
            with st.spinner("Analisando a fotografia…"):
                material, escore = classificar(rede_em_cache(), imagem)
        except Exception:
            logging.exception("Falha ao carregar ou executar o classificador")
            st.error("Classificador indisponível. O primeiro uso precisa de internet para baixar o modelo.")
            if st.button("Tentar novamente"):
                st.rerun()
            return

        if escore < LIMIAR:
            st.subheader("Identificação inconclusiva")
            st.write("Tente outra foto com um único objeto, mais perto e em um fundo simples.")
        elif material is None:
            st.subheader("Fora das categorias")
            st.write("A rede associou a imagem a um resíduo fora dos cinco materiais da triagem.")
        else:
            st.subheader(material)
            st.write(f"Escore do modelo · {escore:.0%}")


st.html(RAIZ / "interface.css")
amostra = st.session_state.get("amostra")
inicio = amostra is None and st.session_state.get("fotografia") is None
exibir_abertura(inicio)

with st.container(key="ferramenta"):
    st.header("Analisar uma foto", anchor="analisar")
    st.write("Enquadre um único objeto e use boa iluminação. Envie até 12 MB e 24 megapixels.")
    foto = st.file_uploader(
        "Selecionar fotografia",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        max_upload_size=12,
        on_change=selecionar_amostra,
        key="fotografia",
    )
    if foto is None and amostra is None:
        exibir_galeria()
    else:
        if foto is None:
            foto = BytesIO((RAIZ / "amostras" / AMOSTRAS[amostra]).read_bytes())
            st.button("Trocar amostra", on_click=selecionar_amostra)
        exibir_resultado(foto)
