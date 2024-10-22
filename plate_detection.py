#!/usr/bin/env python3
"""
Este script realiza la detección del número de placa en una imagen de entrada
y utiliza OCR para extraer los caracteres de la placa.

Authors:
Diego (A01643382)
Gabo (A01642991)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import argparse

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesa la imagen para detectar la placa.

    Parámetros:
        image (np.ndarray): Imagen de entrada como matriz NumPy.
    
    Regresa:
        np.ndarray: Imagen preprocesada con la placa resaltada.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

    # Aplicar un filtro bilateral para reducir el ruido mientras se conservan los bordes
    filtered_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Aplicar detección de bordes usando Canny
    edges = cv2.Canny(filtered_image, 30, 200)

    # Aplicar una transformación morfológica para cerrar pequeños espacios entre contornos
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return morph_image