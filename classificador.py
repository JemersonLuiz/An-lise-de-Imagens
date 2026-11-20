from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

MODELO = "SriramRokkam/wastewise-garbage-cls"
REVISAO = "a30c36c6b181ac0d2eb387bbd4f6d4a5b88ee078"
CLIP = "Xenova/clip-vit-base-patch32"
REVISAO_CLIP = "6b7c523703ae26e39be4b03774966f8571c7f523"
REFERENCIAS = Path(__file__).with_name("referencias_clip.npz")
# Ordem das oito saídas do modelo nesta revisão. Papel inclui papelão.
MATERIAIS = (None, "Orgânico", "Papel", "Vidro", "Metal", "Papel", "Plástico", None)
LIMIAR = 0.65
SIMILARIDADE_MINIMA = 0.22
MARGEM_MINIMA = 0.012
FORMATOS = ("JPEG", "PNG", "WEBP", "BMP")


@dataclass(frozen=True)
class Redes:
    materiais: ort.InferenceSession
    contexto: ort.InferenceSession
    residuos: np.ndarray
    outros: np.ndarray


def carregar_rede() -> Redes:
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = 2

    def sessao(modelo, arquivo, revisao):
        caminho = hf_hub_download(modelo, arquivo, revision=revisao)
        return ort.InferenceSession(caminho, opcoes, providers=["CPUExecutionProvider"])

    with np.load(REFERENCIAS, allow_pickle=False) as textos:
        return Redes(sessao(MODELO, "wastewise-yolo.onnx", REVISAO),
                     sessao(CLIP, "onnx/vision_model_quantized.onnx", REVISAO_CLIP),
                     textos["residuos"], textos["outros"])


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


def parece_residuo(redes: Redes, imagem: Image.Image) -> bool:
    # Preserva a cena inteira. Um recorte central pode remover o objeto principal.
    quadro = ImageOps.pad(imagem, (224, 224), method=Image.Resampling.BICUBIC,
                          color=(123, 117, 104))
    pixels = np.asarray(quadro, dtype=np.float32) / 255.0
    # Média e desvio do pré-processador publicado com esta revisão do CLIP.
    media = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    desvio = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    entrada = ((pixels - media) / desvio).transpose(2, 0, 1)[None, ...]
    vetor = redes.contexto.run(["image_embeds"], {"pixel_values": entrada})[0][0]
    vetor = vetor / np.linalg.norm(vetor)
    positivo = np.max(redes.residuos @ vetor)
    negativo = np.max(redes.outros @ vetor)
    return bool(positivo >= SIMILARIDADE_MINIMA and positivo - negativo >= MARGEM_MINIMA)


def classificar(redes: Redes, imagem: Image.Image) -> tuple[str | None, float | None]:
    if not parece_residuo(redes, imagem):
        return None, None
    rede = redes.materiais
    reduzida = imagem.resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(reduzida, dtype=np.float32) / 255.0
    entrada = pixels.transpose(2, 0, 1)[None, ...]
    saida = rede.run(None, {rede.get_inputs()[0].name: entrada})[0][0]
    escores = {}
    for material, valor in zip(MATERIAIS, saida, strict=True):
        escores[material] = escores.get(material, 0.0) + float(valor)
    material = max(escores, key=escores.get)
    return material, escores[material]
