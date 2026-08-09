from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.fits_loader import cargar_fits

from src.segmentacion.sam_segmentacion import (
    cargar_modelo,
    preparar_imagen,
    generar_mascaras
)


ARCHIVO_FITS = Path(
    "data/AS209_continuum.fits"
)

CHECKPOINT_SAM = Path(
    "models/sam_vit_b_01ec64.pth"
)


def preparar_visualizacion(imagen):
    """
    Ajusta el contraste de la imagen para visualizar
    mejor las estructuras astronómicas.
    """

    imagen = np.asarray(
        imagen,
        dtype=np.float32
    )

    # Eliminar valores no válidos
    imagen = np.nan_to_num(
        imagen,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Percentiles para mejorar el contraste
    minimo = np.percentile(
        imagen,
        1
    )

    maximo = np.percentile(
        imagen,
        99
    )

    imagen = np.clip(
        imagen,
        minimo,
        maximo
    )

    # Normalizar entre 0 y 1
    imagen = (
        imagen - minimo
    ) / (
        maximo - minimo
    )

    return imagen


def main():

    # --------------------------------
    # 1. Configurar dispositivo
    # --------------------------------

    # Por ahora utilizaremos CPU.
    device = torch.device("cpu")

    print("--------------------------------")
    print("PRUEBA SAM - AS209")
    print("--------------------------------")

    print(
        f"Dispositivo utilizado: {device}"
    )


    # --------------------------------
    # 2. Cargar FITS
    # --------------------------------

    print("\nCargando FITS...")

    imagen = cargar_fits(
        ARCHIVO_FITS
    )


    # --------------------------------
    # 3. Preparar imagen para SAM
    # --------------------------------

    print("\nPreparando imagen...")

    imagen_rgb = preparar_imagen(
        imagen
    )

    print(
        "Forma de la imagen para SAM:",
        imagen_rgb.shape
    )

    print(
        "Tipo de datos:",
        imagen_rgb.dtype
    )


    # --------------------------------
    # 4. Cargar SAM
    # --------------------------------

    print("\nCargando SAM...")

    predictor = cargar_modelo(
        CHECKPOINT_SAM,
        device
    )


    # --------------------------------
    # 5. Generar máscaras
    # --------------------------------

    print("\nGenerando máscaras con SAM...")

    masks = generar_mascaras(
        predictor,
        imagen_rgb
    )

    print(
        f"Cantidad de máscaras generadas: {len(masks)}"
    )


    # --------------------------------
    # 6. Mostrar información de máscaras
    # --------------------------------

    for i, mask in enumerate(masks):

        print(
            f"\nMáscara {i + 1}"
        )

        print(
            "Área:",
            mask["area"]
        )

        print(
            "IoU predicho:",
            mask["predicted_iou"]
        )

        print(
            "Estabilidad:",
            mask["stability_score"]
        )

        print(
            "Bounding box:",
            mask["bbox"]
        )


    # --------------------------------
    # 7. Preparar imagen para visualizar
    # --------------------------------

    imagen_visual = preparar_visualizacion(
        imagen
    )


    # --------------------------------
    # 8. Visualización
    # --------------------------------

    cantidad_mascaras = len(masks)

    cantidad_imagenes = cantidad_mascaras + 1

    columnas = 3

    filas = int(
        np.ceil(
            cantidad_imagenes / columnas
        )
    )

    plt.figure(
        figsize=(15, 5 * filas)
    )


    # --------------------------------
    # Imagen original
    # --------------------------------

    plt.subplot(
        filas,
        columnas,
        1
    )

    plt.imshow(
        imagen_visual,
        cmap="gray",
        origin="lower"
    )

    plt.title(
        "AS209"
    )

    plt.axis("off")


    # --------------------------------
    # Mostrar cada máscara
    # --------------------------------

    for i, mask in enumerate(masks):

        plt.subplot(
            filas,
            columnas,
            i + 2
        )

        plt.imshow(
            imagen_visual,
            cmap="gray",
            origin="lower"
        )

        segmentacion = mask["segmentation"]

        plt.contour(
            segmentacion,
            levels=[0.5]
        )

        plt.title(
            f"Máscara {i + 1}"
        )

        plt.axis("off")


    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()