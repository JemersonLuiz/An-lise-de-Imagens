"""Regenera os vetores de texto do filtro. Requer tokenizers apenas nesta etapa."""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from classificador import CLIP, REFERENCIAS, REVISAO_CLIP


DESCRICOES = {
    "residuos": [
        "a discarded metal food can", "a crushed aluminum drink can", "a piece of scrap metal",
        "an empty plastic water bottle", "a plastic bag or plastic wrapper", "discarded plastic packaging",
        "an empty glass bottle", "a glass jar for recycling", "broken pieces of glass",
        "a sheet of waste paper", "a folded newspaper", "a crumpled piece of paper",
        "a cardboard box", "a piece of corrugated cardboard",
        "fruit peels and vegetable scraps", "a discarded banana peel", "leftover food waste",
        "a discarded battery", "a piece of litter ready for recycling",
        "a beverage can with a printed label", "a piece of aluminum foil",
        "a disposable plastic food container", "an empty plastic cup",
        "a glass beverage bottle lying on its side", "a newspaper or discarded magazine",
        "a used paper envelope", "a flattened cardboard package",
    ],
    "outros": [
        "a portrait photograph of a person", "a selfie", "a group of people at an event",
        "a person indoors", "a photograph of a pet", "a wild animal", "a landscape in nature",
        "a street with cars and buildings", "the interior of a room", "a living plant or flower",
        "a meal served on a plate", "a computer or electronic device", "a screenshot of a website",
        "a piece of clothing", "a piece of furniture", "a vehicle", "a toy", "a painting or drawing",
        "a blank solid color image", "random noise or a blurred photograph",
    ],
}


def gerar():
    def baixar(nome):
        return hf_hub_download(CLIP, nome, revision=REVISAO_CLIP)

    tokenizer = Tokenizer.from_file(baixar("tokenizer.json"))
    modelo = ort.InferenceSession(baixar("onnx/text_model_quantized.onnx"),
                                  providers=["CPUExecutionProvider"])
    referencias = {}
    for grupo, descricoes in DESCRICOES.items():
        linhas = []
        for descricao in descricoes:
            ids = np.array([tokenizer.encode(descricao).ids], dtype=np.int64)
            vetor = modelo.run(["text_embeds"], {"input_ids": ids})[0][0]
            linhas.append(vetor / np.linalg.norm(vetor))
        referencias[grupo] = np.stack(linhas)
    np.savez_compressed(REFERENCIAS, **referencias)


if __name__ == "__main__":
    gerar()
