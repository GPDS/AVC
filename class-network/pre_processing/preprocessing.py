import sys
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt


class PreProcessing:
    def __init__(self, **kwargs):
        self.files = [] if "files" not in kwargs else kwargs["files"]
        self.images = []

    def read_img(self, janelamento = False):
        for file in self.files:
            img_dc = pydicom.dcmread(file)
            if janelamento:
                img_dc = self.make_jan(img_dc, janelamento=janelamento)
            self.images.append(img_dc)

    @staticmethod
    def make_jan(image, janelamento: list):
        """Janelamento : Lista com o valor mínimo e máximo de janela, transforma esse valor em um array numpy"""
        min_valor = min(janelamento)
        max_valor = max(janelamento)
        imagem_janelada = np.copy(image)
        imagem_janelada[image < min_valor] = min_valor
        imagem_janelada[image > max_valor] = max_valor
        return imagem_janelada

