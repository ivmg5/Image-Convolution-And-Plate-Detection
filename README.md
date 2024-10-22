# Proyecto de Convolución y Detección de Placas

Este repositorio contiene dos scripts principales que realizan operaciones de procesamiento de imágenes. El primer script realiza una convolución sobre imágenes utilizando filtros (como Sobel y Gaussian), y el segundo script detecta el número de placa en una imagen y utiliza OCR para extraer los caracteres.

## Estructura del proyecto

- convolution.py: Este script aplica operaciones de convolución a una imagen de entrada utilizando filtros como Sobel y Gaussian, y admite diferentes tipos de padding.
- plate_detection.py: Este script detecta una placa en una imagen de entrada y utiliza OCR para extraer los caracteres.
- Turquia.jpg: Imagen de prueba utilizada con el script convolution.py.
- Placa.jpg: Imagen de prueba utilizada con el script plate_detection.py.

## Requisitos

Para ejecutar estos scripts, necesitarás instalar las siguientes dependencias:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pytesseract

Puedes instalar las dependencias utilizando pip:

bash
pip install opencv-python-headless numpy matplotlib pytesseract


## Uso

### 1. Convolución en una Imagen (convolution.py)

Este script aplica un filtro de convolución a una imagen de entrada. Los filtros disponibles son Sobel (para detección de bordes) y Gaussian (para suavizado).

#### Ejemplo de ejecución

bash
python convolution.py -i Turquia.jpg -f sobel -p same -k 3 -s 1.0


- -i: Ruta a la imagen de entrada.
- -f: Filtro a aplicar (gaussian o sobel).
- -p: Tipo de padding (valid o same).
- -k: Tamaño del kernel.
- -s: Valor de sigma para el kernel.

### 2. Detección de Placas y OCR (plate_detection.py)

Este script detecta la placa de un vehículo en una imagen y extrae los caracteres utilizando OCR.

#### Ejemplo de ejecución

bash
python plate_detection.py -i Placa.jpg


- -i: Ruta a la imagen de entrada.

## Justificación de los Filtros Escogidos

### 1. Filtro Gaussiano
El *filtro Gaussiano* es un filtro de suavizado que se utiliza comúnmente para reducir el ruido en las imágenes. Al aplicar un filtro Gaussiano, se promedian los valores de los píxeles en función de su proximidad, lo que resulta en una imagen más suave y menos propensa a pequeñas variaciones causadas por el ruido. Esto es especialmente útil antes de aplicar otros algoritmos que puedan verse afectados por el ruido, como la detección de bordes o el análisis de patrones.

*Motivación de la elección*:
- *Reducción de ruido*: El filtro Gaussiano es ideal para eliminar el ruido en la imagen sin perder demasiada información estructural importante.
- *Preparación para otras operaciones*: Al suavizar una imagen, facilita la posterior detección de bordes y la mejora de la claridad en zonas de bajo contraste.
- *Aplicación generalizada*: Se utiliza comúnmente en la preprocesamiento de imágenes en sistemas de visión artificial y otras aplicaciones de análisis visual.

### 2. Filtro Sobel
El *filtro Sobel* es un filtro de detección de bordes que resalta las zonas de una imagen donde ocurre un cambio abrupto en la intensidad de los píxeles, lo que corresponde a los bordes o límites de los objetos. Este filtro es ampliamente utilizado para la detección de contornos, ya que calcula los gradientes de intensidad en la imagen.

*Motivación de la elección*:
- *Detección de bordes*: El filtro Sobel permite identificar con claridad los contornos de objetos dentro de la imagen, lo que es útil en una variedad de aplicaciones, desde la segmentación de imágenes hasta el reconocimiento de objetos.
- *Facilidad de uso*: Es un filtro computacionalmente eficiente y sencillo de aplicar, lo que lo hace apropiado para análisis rápidos de imágenes.
- *Enriquecimiento de detalles*: Aunque resalta los bordes, también conserva las estructuras fundamentales de los objetos, permitiendo un análisis más preciso cuando es necesario identificar los bordes como características clave de la imagen.

### Comparación y Propósito de Uso

- *Filtro Gaussiano*: Se emplea para suavizar la imagen, reducir el ruido, y hacer que los detalles menores no interfieran con el análisis principal.
- *Filtro Sobel*: Ideal para extraer bordes y contornos, lo que permite identificar formas y objetos en la imagen con mayor claridad.

Ambos filtros fueron seleccionados debido a su efectividad y su rol complementario: el filtro Gaussiano para preparar la imagen eliminando el ruido, y el filtro Sobel para detectar los bordes con mayor precisión. Este enfoque garantiza una mejor calidad de procesamiento, especialmente en imágenes donde se requiere claridad en los contornos para un análisis más profundo o la extracción de características, como en el caso de la detección de placas vehiculares. 

## Modificación de Parámetros en convolution.py

### Parámetro -k (Tamaño del Kernel)

El tamaño del kernel (-k) determina el área de la imagen que se considera durante la operación de convolución. El kernel es una pequeña matriz que se desplaza sobre la imagen para aplicar la transformación, como suavizado o detección de bordes.

- **Si -k es mayor*: Un kernel más grande genera un efecto más fuerte. Para el **filtro Gaussiano, produce un mayor desenfoque. Para el **filtro Sobel*, los bordes se ven menos definidos.
- **Si -k es menor**: Un kernel más pequeño genera menos suavizado (filtro Gaussiano) o bordes más nítidos (filtro Sobel).

### Parámetro -s (Sigma)

El parámetro -s controla el valor de la desviación estándar (sigma) en la función gaussiana.

- **Si -s es mayor*: Un valor más alto genera un mayor suavizado en el **filtro Gaussiano, y bordes más difusos en el **filtro Sobel*.
- **Si -s es menor*: Un valor más bajo preserva más detalles en el **filtro Gaussiano* y genera bordes más definidos en el *filtro Sobel*.

### Parámetro -p (Padding)

El parámetro -p define el tipo de padding utilizado durante la convolución.

- **Si -p es same**: La imagen de salida tendrá el mismo tamaño que la original, pero los bordes pueden verse afectados por artefactos.
- **Si -p es valid**: La imagen de salida será más pequeña, pero sin artefactos en los bordes.

## Autores

- Diego (A01643382)
- Gabo (A01642991)