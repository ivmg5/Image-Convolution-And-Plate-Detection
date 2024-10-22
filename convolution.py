#!/usr/bin/env python3
"""
Este script realiza una operación de convolución en una imagen de entrada utilizando diferentes filtros
(Sobel y Gaussian) y admite el padding para controlar el tamaño de la salida.

Authors:
Diego (A01643382)
Gabo (A01642991)
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(image: np.ndarray, kernel: np.ndarray, padding: str = 'valid') -> np.ndarray:
    """
    Aplica una operación de convolución a una imagen utilizando el kernel dado y el tipo de padding especificado.

    Parámetros:
        image (np.ndarray): Imagen de entrada como matriz NumPy (puede ser en color o en escala de grises).
        kernel (np.ndarray): Kernel de convolución como matriz NumPy.
        padding (str): Tipo de padding ('valid' para no rellenar, 'same' para mantener el mismo tamaño).

    Regresa:
        np.ndarray: El resultado de la operación de convolución.
    """
    # Verificar si la imagen tiene múltiples canales (imagen en color)
    if len(image.shape) == 3:
        # Dividir la imagen en sus canales de color
        channels = cv2.split(image)
        convolved_channels = []
        # Aplicar convolución a cada canal
        for channel in channels:
            convolved_channel = convolution_2d(channel, kernel, padding)
            convolved_channels.append(convolved_channel)
        # Combinar los canales nuevamente en una imagen en color
        convolved_image = cv2.merge(convolved_channels)
    else:
        # La imagen es en escala de grises; aplicar convolución directamente
        convolved_image = convolution_2d(image, kernel, padding)
    return convolved_image