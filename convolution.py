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

def convolution_2d(image: np.ndarray, kernel: np.ndarray, padding: str) -> np.ndarray:
    """
    Realiza una operación de convolución en una imagen 2D (escala de grises) con el tipo de padding especificado.

    Parámetros:
        image (np.ndarray): Imagen de entrada 2D como matriz NumPy.
        kernel (np.ndarray): Kernel de convolución 2D como matriz NumPy.
        padding (str): Tipo de padding ('valid' para no rellenar, 'same' para mantener el mismo tamaño).

    Regresa:
        np.ndarray: El resultado de la operación de convolución.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calcular el tamaño del padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    if padding == 'same':
        # Agregar padding de ceros alrededor de la imagen
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                              mode='constant', constant_values=0)
        output_height = image_height
        output_width = image_width
    elif padding == 'valid':
        # Sin padding
        padded_image = image
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
    else:
        raise ValueError("Unsupported padding type. Use 'valid' or 'same'.")

    # Inicializar la imagen de salida
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Realizar la operación de convolución
    for row in range(output_height):
        for col in range(output_width):
            # Extraer la región de interés
            region = padded_image[row:row + kernel_height, col:col + kernel_width]
            # Multiplicación elemento a elemento y suma
            output[row, col] = np.sum(region * kernel)

    # Limitar los valores para que estén dentro del rango válido
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)