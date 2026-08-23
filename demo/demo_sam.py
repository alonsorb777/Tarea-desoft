from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

from src.fits_loader import cargar_fits

from src.segmentacion.sam_segmentacion import (
    segmentar_objeto,
    ordenar_mascaras,
)

# Connfiguracion


OBJETIVOS_DISPONIBLES = [
    "AS209",
    "HD163296"
]

OBJETIVO = sys.argv[1] if len(sys.argv) > 1 else "AS209"

if OBJETIVO not in OBJETIVOS_DISPONIBLES:
    raise ValueError(
        f"Objeto no disponible: {OBJETIVO}. "
        f"Opciones disponibles: {', '.join(OBJETIVOS_DISPONIBLES)}"
    )

ARCHIVO_FITS = Path(
    f"data/{OBJETIVO}_continuum.fits"
)

if not ARCHIVO_FITS.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo FITS de {OBJETIVO}: "
        f"{ARCHIVO_FITS}"
    )

CHECKPOINT_SAM = Path(
    "models/sam_vit_b_01ec64.pth"
)


# Preparar imagen para visualización


def preparar_visualizacion(imagen):
    """
    Prepara una imagen FITS para visualizarla
    correctamente.

    Se utiliza la misma normalización general
    empleada para preparar la imagen para SAM.
    """

    imagen = np.asarray(
        imagen,
        dtype=np.float32
    )

    imagen = np.nan_to_num(
        imagen,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    minimo = np.percentile(
        imagen,
        1
    )

    maximo = np.percentile(
        imagen,
        99.5
    )

    if maximo <= minimo:
        raise ValueError(
            "No es posible visualizar la imagen."
        )

    imagen = np.clip(
        imagen,
        minimo,
        maximo
    )

    imagen = (
        imagen - minimo
    ) / (
        maximo - minimo
    )

    imagen = (
        np.arcsinh(
            10 * imagen
        )
        / np.arcsinh(10)
    )

    return imagen



# Visualizar máscaras


def visualizar_resultados(
    imagen,
    masks,
    cantidad=10
):
    """
    Muestra la imagen original y las mejores
    máscaras generadas por SAM.
    """

    imagen_visual = preparar_visualizacion(
        imagen
    )

    masks_ordenadas = ordenar_mascaras(
        masks,
        criterio="predicted_iou"
    )

    mejores = masks_ordenadas[
        :cantidad
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )

   
    # Imagen original


    axes[0].imshow(
        imagen_visual,
        cmap="inferno",
        origin="lower"
    )

    axes[0].set_title(
        f"{OBJETIVO} - Imagen DSHARP"
    )

    axes[0].set_xlabel(
        "Píxeles"
    )

    axes[0].set_ylabel(
        "Píxeles"
    )

  
    # Máscaras


    axes[1].imshow(
        imagen_visual,
        cmap="inferno",
        origin="lower"
    )

    for i, mask in enumerate(
        mejores
    ):

        axes[1].contour(
            mask["segmentation"],
            levels=[0.5],
            linewidths=1
        )

    axes[1].set_title(
        f"Top {len(mejores)} máscaras SAM"
    )

    axes[1].set_xlabel(
        "Píxeles"
    )

    axes[1].set_ylabel(
        "Píxeles"
    )

    plt.tight_layout()

    plt.show()



# Informacion de mascaras


def mostrar_informacion_mascaras(
    masks,
    cantidad=10
):
    """
    Muestra información de las mejores máscaras.
    """

    masks_ordenadas = ordenar_mascaras(
        masks,
        criterio="predicted_iou"
    )

    mejores = masks_ordenadas[
        :cantidad
    ]

    print()
    print("=" * 60)
    print("INFORMACIÓN DE LAS MÁSCARAS")
    print("=" * 60)

    for i, mask in enumerate(
        mejores
    ):

        print()
        print(
            f"Máscara {i + 1}"
        )

        print(
            f"Área: {mask['area']}"
        )

        print(
            f"IoU predicho: "
            f"{mask['predicted_iou']:.4f}"
        )

        print(
            f"Estabilidad: "
            f"{mask['stability_score']:.4f}"
        )

        print(
            f"Bounding box: "
            f"{mask['bbox']}"
        )



# Main


def main():

    print("=" * 60)
    print(f"DEMO DE SEGMENTACIÓN SAM - {OBJETIVO}")
    print("=" * 60)


    # 1. Seleccionar dispositivo
  

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Dispositivo utilizado: {device}"
    )


    # 2. Cargar imagen FITS


    print()
    print("Cargando imagen FITS...")

    imagen = cargar_fits(
        ARCHIVO_FITS
    )

    print(
        f"Forma de la imagen: "
        f"{imagen.shape}"
    )


    # 3. Ejecutar segmentación automática

    print()
    print(
        "Ejecutando segmentación automática..."
    )

    masks, imagen_sam, escala = segmentar_objeto(
        imagen,
        CHECKPOINT_SAM,
        device
    )

    print(
        f"Forma utilizada por SAM: "
        f"{imagen_sam.shape}"
    )

    print(
        f"Escala utilizada: "
        f"{escala:.4f}"
    )
   
    # 4. Información


    mostrar_informacion_mascaras(
        masks,
        cantidad=10
    )


    # 5. Visualización


    print()
    print(
        "Abriendo visualización..."
    )

    visualizar_resultados(
        imagen_sam,
        masks,
        cantidad=10
    )

# Ejecución


if __name__ == "__main__":
    main()