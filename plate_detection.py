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

def detect_plate(image: np.ndarray, processed_image: np.ndarray) -> np.ndarray:
    """
    Detecta la placa del vehículo en la imagen.

    Parámetros:
        image (np.ndarray): Imagen original.
        processed_image (np.ndarray): Imagen preprocesada (binarizada).

    Regresa:
        np.ndarray: La región de la imagen que contiene la placa.
    """
    # Encontrar los contornos en la imagen procesada
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos por área para identificar la placa (asumiendo que es uno de los más grandes)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour = None

    for contour in contours:
        # Aproximar la forma del contorno
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si el contorno tiene 4 lados, es probable que sea la placa
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        # Obtener el rectángulo delimitador de la placa
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = image[y:y+h, x:x+w]
        return plate_image
    else:
        return None

def ocr_plate(plate_image: np.ndarray) -> str:
    """
    Extrae los caracteres de la placa utilizando OCR.

    Parámetros:
        plate_image (np.ndarray): Imagen de la placa recortada.
    
    Regresa:
        str: Texto extraído de la placa.
    """
    # Convertir la imagen a escala de grises para mejorar el OCR
    gray_plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización para aumentar el contraste
    _, threshold_plate_image = cv2.threshold(gray_plate_image, 150, 255, cv2.THRESH_BINARY)

    # Usar OCR para extraer el texto
    config = '--psm 8'  # PSM 8 trata cada palabra como una línea de texto
    plate_text = pytesseract.image_to_string(threshold_plate_image, config=config)

    return plate_text.strip()

def main():
    """
    Función principal para detectar la placa y extraer los caracteres.
    """
    parser = argparse.ArgumentParser(description="Detecta la placa de un vehículo y extrae sus caracteres.")
    parser.add_argument("-i", "--image", required=True, help="Ruta a la imagen de entrada.")
    args = parser.parse_args()

    # Leer la imagen de entrada
    image = cv2.imread(args.image)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Detectar la placa
    plate_image = detect_plate(image, processed_image)

    if plate_image is not None:
        # Extraer texto de la placa
        plate_text = ocr_plate(plate_image)

        # Mostrar la imagen de la placa recortada con los caracteres detectados en el título
        plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Placa Detectada: {plate_text}")
        plt.axis('off')
        plt.show()

        print(f"Texto detectado en la placa: {plate_text}")
    else:
        print("No se pudo detectar una placa en la imagen.")

if __name__ == "_main_":
    main()
