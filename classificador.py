from typing import BinaryIO

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

MODELO = "SriramRokkam/wastewise-garbage-cls"
REVISAO = "a30c36c6b181ac0d2eb387bbd4f6d4a5b88ee078"
# Ordem das oito saídas do modelo nesta revisão. Papel inclui papelão.
MATERIAIS = (None, "Orgânico", "Papel", "Vidro", "Metal", "Papel", "Plástico", None)
LIMIAR = 0.65
FORMATOS = ("JPEG", "PNG", "WEBP", "BMP")


def carregar_rede() -> ort.InferenceSession:
    caminho = hf_hub_download(MODELO, "wastewise-yolo.onnx", revision=REVISAO)
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = 2
    return ort.InferenceSession(caminho, opcoes, providers=["CPUExecutionProvider"])


def ler_foto(arquivo: BinaryIO) -> Image.Image:
    arquivo.seek(0)
    with Image.open(arquivo, formats=FORMATOS) as original:
        if original.width * original.height > 24_000_000:
            raise ValueError("A foto excede 24 megapixels. Reduza a resolução e tente novamente.")
        imagem = ImageOps.exif_transpose(original)
        if imagem.has_transparency_data:
            fundo = Image.new("RGB", imagem.size, "white")
            rgba = imagem.convert("RGBA")
            fundo.paste(rgba, mask=rgba.getchannel("A"))
            return fundo
        return imagem.convert("RGB")


def classificar(rede: ort.InferenceSession, imagem: Image.Image) -> tuple[str | None, float]:
    reduzida = imagem.resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(reduzida, dtype=np.float32) / 255.0
    entrada = pixels.transpose(2, 0, 1)[None, ...]
    saida = rede.run(None, {rede.get_inputs()[0].name: entrada})[0][0]
    escores = {}
    for material, valor in zip(MATERIAIS, saida, strict=True):
        escores[material] = escores.get(material, 0.0) + float(valor)
    material = max(escores, key=escores.get)
    return material, escores[material]
