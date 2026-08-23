from pathlib import Path
from urllib.request import urlretrieve
import os
import urllib.request

import numpy as np

from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator
)


# Configuración


SAM_URL = (
    "https://dl.fbaipublicfiles.com/"
    "segment_anything/sam_vit_b_01ec64.pth"
)



# Checkpoint

def obtener_checkpoint(path_checkpoint):
    # 1. Obtener la carpeta donde debe guardarse el archivo
    folder = os.path.dirname(path_checkpoint)
    
    # 2. Si la ruta incluye una carpeta y no existe, la crea
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    # 3. Si el archivo .pth no existe, lo descarga automáticamente
    if not os.path.exists(path_checkpoint):
        print(f"Descargando checkpoint SAM en: {path_checkpoint}...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        
        # Descarga con User-Agent para evitar bloqueos HTTP 403
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(path_checkpoint, 'wb') as out_file:
            out_file.write(response.read())
            
        print("¡Descarga de checkpoint completada con éxito!")
        
    return path_checkpoint

    print("\nNo se encontró el modelo SAM ViT-B.")

    respuesta = input(
        "¿Deseas descargarlo ahora? [s/n]: "
    ).strip().lower()

    if respuesta not in ["s", "si", "sí"]:
        raise FileNotFoundError(
            "No se puede ejecutar SAM sin el checkpoint."
        )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nDescargando SAM ViT-B...")
    print(f"Destino: {checkpoint_path}")

    urllib.request.urlretrieve(
        SAM_URL,
        checkpoint_path
    )

    print(
        "\nModelo SAM descargado correctamente."
    )

    return checkpoint_path



# Cargar modelo


def cargar_modelo(checkpoint_path, device):
    """
    Carga el modelo SAM ViT-B.

    Retorna:
        SamPredictor
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo SAM: "
            f"{checkpoint_path}"
        )

    print("\nCargando modelo SAM ViT-B...")

    sam = sam_model_registry["vit_b"](
        checkpoint=str(checkpoint_path)
    )

    sam.to(
        device=device
    )

    predictor = SamPredictor(
        sam
    )

    print(
        "Modelo SAM cargado correctamente."
    )

    return predictor



# Preparar imagen


def preparar_imagen(
    imagen,
    percentil_min=1.0,
    percentil_max=99.5,
    asinh_scale=10.0
):
    """
    Convierte una imagen científica 2D
    en una imagen RGB uint8 compatible con SAM.

    Se utiliza normalización por percentiles
    y transformación asinh para mejorar
    la visualización de estructuras débiles.
    """

    imagen = np.asarray(
        imagen,
        dtype=np.float32
    )

    # Eliminar valores inválidos
    imagen = np.nan_to_num(
        imagen,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Calcular percentiles
    minimo = np.percentile(
        imagen,
        percentil_min
    )

    maximo = np.percentile(
        imagen,
        percentil_max
    )

    if maximo <= minimo:
        raise ValueError(
            "No es posible normalizar la imagen: "
            "el máximo debe ser mayor que el mínimo."
        )

    # Recortar valores extremos
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

    # Transformación asinh
    imagen = (
        np.arcsinh(
            asinh_scale * imagen
        )
        / np.arcsinh(
            asinh_scale
        )
    )

    # Convertir a uint8
    imagen = (
        imagen * 255
    ).astype(
        np.uint8
    )

    # Convertir escala de grises a RGB
    imagen_rgb = np.stack(
        [
            imagen,
            imagen,
            imagen
        ],
        axis=-1
    )

    return imagen_rgb



# Reducir imagen


def reducir_imagen_para_sam(
    imagen,
    max_size=1500
):
    """
    Reduce el tamaño de una imagen manteniendo
    su proporción.
    """

    import cv2

    imagen = np.asarray(
        imagen
    )

    altura, ancho = imagen.shape[:2]

    escala = min(
        max_size / ancho,
        max_size / altura
    )

    # Si ya es suficientemente pequeña,
    # no modificarla.
    if escala >= 1:
        return (
            imagen,
            1.0
        )

    nuevo_ancho = int(
        ancho * escala
    )

    nueva_altura = int(
        altura * escala
    )

    imagen_reducida = cv2.resize(
        imagen,
        (
            nuevo_ancho,
            nueva_altura
        ),
        interpolation=cv2.INTER_AREA
    )

    print(
        f"Imagen reducida para SAM: "
        f"{ancho}x{altura} -> "
        f"{nuevo_ancho}x{nueva_altura}"
    )

    return (
        imagen_reducida,
        escala
    )



# Convertir coordenadas


def convertir_coordenadas(
    x,
    y,
    escala
):
    """
    Convierte coordenadas de la imagen original
    a coordenadas de la imagen utilizada por SAM.
    """

    if escala <= 0:
        raise ValueError(
            "La escala debe ser mayor que cero."
        )

    return [
        x * escala,
        y * escala
    ]



# Segmentacion interactiva


def segmentar_disco(
    predictor,
    imagen,
    puntos,
    etiquetas
):
    """
    Segmenta una estructura utilizando SAM.

    """

    if len(puntos) == 0:
        raise ValueError(
            "Debe existir al menos un punto."
        )

    if len(puntos) != len(etiquetas):
        raise ValueError(
            "La cantidad de puntos y etiquetas "
            "debe ser igual."
        )

    puntos = np.asarray(
        puntos,
        dtype=np.float32
    )

    etiquetas = np.asarray(
        etiquetas,
        dtype=np.int32
    )

    # Las etiquetas permitidas por SAM
    # son 0 (negativo) y 1 (positivo).
    if not np.all(
        np.isin(
            etiquetas,
            [0, 1]
        )
    ):
        raise ValueError(
            "Las etiquetas deben ser 0 o 1."
        )

    # Preparar imagen para SAM
    predictor.set_image(
        imagen
    )

    # Ejecutar predicción
    masks, scores, _ = predictor.predict(
        point_coords=puntos,
        point_labels=etiquetas,
        multimask_output=True
    )

    # Seleccionar la máscara con mejor score
    mejor_indice = int(
        np.argmax(scores)
    )

    print(
        "\nScores de SAM:",
        scores
    )

    print(
        "Mejor score:",
        scores[mejor_indice]
    )

    return {
        "mask": masks[mejor_indice],
        "score": float(
            scores[mejor_indice]
        ),
        "all_masks": masks,
        "all_scores": scores
    }


# ==========================================================
# SEGMENTACIÓN AUTOMÁTICA
# ==========================================================

def generar_mascaras(
    predictor,
    imagen,
    points_per_side=32,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.75,
    min_mask_region_area=100,
    crop_n_layers=0
):
    """
    Genera máscaras automáticamente utilizando SAM.

    Esta función permite explorar automáticamente
    las estructuras presentes en la imagen.
    """

    mask_generator = SamAutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers
    )

    print(
        "\nGenerando máscaras automáticamente..."
    )

    masks = mask_generator.generate(
        imagen
    )

    print(
        f"SAM generó {len(masks)} máscaras."
    )

    return masks



# Ordenar máscaras


def ordenar_mascaras(
    masks,
    criterio="predicted_iou"
):
    """
    Ordena las máscaras según un criterio.

    Criterios disponibles:
        predicted_iou
        stability_score
        area
    """

    if criterio == "predicted_iou":

        return sorted(
            masks,
            key=lambda x: x["predicted_iou"],
            reverse=True
        )

    elif criterio == "stability_score":

        return sorted(
            masks,
            key=lambda x: x["stability_score"],
            reverse=True
        )

    elif criterio == "area":

        return sorted(
            masks,
            key=lambda x: x["area"],
            reverse=True
        )

    else:

        raise ValueError(
            f"Criterio desconocido: {criterio}"
        )