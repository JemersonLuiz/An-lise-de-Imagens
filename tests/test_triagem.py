import unittest
from pathlib import Path

from classificador import LIMIAR, carregar_rede, classificar, ler_foto


class TestTriagem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redes = carregar_rede()

    def prever(self, caminho):
        with caminho.open("rb") as foto:
            return classificar(self.redes, ler_foto(foto))

    def test_retrato_sem_residuo(self):
        # Foto de domínio público da NASA distribuída por scikit-image (astronaut.png).
        caminho = Path(__file__).parent / "fixtures" / "astronauta.jpg"
        self.assertEqual(self.prever(caminho), (None, None))

    def test_amostras_da_galeria(self):
        pasta = Path(__file__).resolve().parents[1] / "amostras"
        for arquivo, esperado in [("metal.jpg", "Metal"), ("papel.jpg", "Papel"), ("vidro.jpg", "Vidro")]:
            with self.subTest(arquivo=arquivo):
                material, escore = self.prever(pasta / arquivo)
                self.assertEqual(material, esperado)
                self.assertGreaterEqual(escore, LIMIAR)
