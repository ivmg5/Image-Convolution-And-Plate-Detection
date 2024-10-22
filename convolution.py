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

def generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Genera un kernetl gaussiano de un tamaño y sigma dados.

    El filtro Gaussiano se utiliza para suavizar la imagen, reduciendo el ruido 
    y los detalles. Esto es útil para eliminar artefactos no deseados y lograr 
    un efecto de desenfoque.

    Parámetros:
        size (int): Tamaño del kernel.
        sigma (float): Desviación típica de la distribución Gaussiana.

    Regresa:
        np.ndarray: Kernel Gaussiano normalizado.
    """
    if size % 2 == 0:
        size += 1

    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)

    kernel = kernel / np.sum(kernel)
    return kernel

def generate_sobel_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Genera un kernel Sobel personalizado con tamaño y sigma ajustables.

    El filtro Sobel se utiliza para la detección de bordes, ya que realza los 
    cambios abruptos en intensidad de píxeles, lo que permite identificar 
    bordes en la imagen.

    Parámetros:
        size (int): Tamaño del kernel (debe ser impar y al menos 3).
        sigma (float): Desviación estándar para el suavizado Gaussiano aplicado antes del cálculo del gradiente.

    Regresa:
        np.ndarray: Kernel Sobel personalizado.
    """
    if size % 2 == 0:
        size += 1
    if size < 3:
        size = 3

    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

    # Calcular el kernel Sobel en X (detectar bordes horizontales)
    sobel_x = -x / (2 * np.pi * sigma*4) * np.exp(-(x**2 + y**2) / (2 * sigma*2))

    sobel_x = sobel_x / np.sum(np.abs(sobel_x))
    return sobel_x

def main():
    """
    Función principal para analizar los argumentos y ejecutar la operación de convolución.
    """
    parser = argparse.ArgumentParser(description="Performs a convolution on an image using different filters and padding.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-f", "--filter", choices=['gaussian', 'sobel'], default='gaussian', help="Filter to apply: 'gaussian' or 'sobel'.")
    parser.add_argument("-p", "--padding", choices=['valid', 'same'], default='valid', help="Padding type: 'valid' (no padding) or 'same' (zero padding).")
    parser.add_argument("-k", "--kernel_size", type=int, default=3, help="Kernel size (must be odd).")
    parser.add_argument("-s", "--sigma", type=float, default=1.0, help="Sigma value for the kernel.")
    args = parser.parse_args()

    # Leer la imagen de entrada
    image = cv2.imread(args.image)

    if image is None:
        print("Error: Image not found or could not be loaded.")
        return

    # Seleccionar el filtro
    if args.filter == 'gaussian':
        kernel = generate_gaussian_kernel(size=args.kernel_size, sigma=args.sigma)
        filter_name = f"Gaussian (Size: {args.kernel_size}, Sigma: {args.sigma})"
        convolved_image = convolution(image, kernel, padding=args.padding)
    elif args.filter == 'sobel':
        kernel = generate_sobel_kernel(size=args.kernel_size, sigma=args.sigma)
        filter_name = f"Sobel (Size: {args.kernel_size}, Sigma: {args.sigma})"
        convolved_image = convolution(image, kernel, padding=args.padding)
    else:
        print("Error: Unsupported filter.")
        return

    # Mostrar las imágenes original y convolucionada
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if args.filter == 'sobel':
        plt.imshow(cv2.cvtColor(convolved_image, cv2.COLOR_BGR2GRAY), cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(convolved_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image with {filter_name}\nPadding: {args.padding}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
