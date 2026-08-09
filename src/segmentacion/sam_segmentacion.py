from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator 

def cargar_modelo(checkpoint_path, device):
    """
    carga el modelo SAM utilizando VIT-b
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo SAM: {checkpoint_path}")

    print(f"Cargando modelo SAM")

    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    sam.to(device=device)

    predictor = SamPredictor(sam)

    print(f"Modelo SAM cargado.")
    return predictor


def generar_mascaras(predictor, imagen):
    """
    Genera máscaras utilizando SAM.
    """

    sam = predictor.model

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100
    )

    print("Generando máscaras con SAM...")

    masks = mask_generator.generate(imagen)

    print(
        f"SAM generó {len(masks)} máscaras."
    )

    return masks

def preparar_imagen(imagen):
    """
    Convierte la imagen 2D del fits a una imagen RGB compatible con SAM.
    """
    imagen = np.asarray(imagen, dtype=np.float32)

    #Reemplaza NaN e infinitos
    imagen = np.nan_to_num(imagen, nan=0.0, posinf=0.0, neginf=0.0)

    #Percentiles para que los valores extremos no dominen la normalización
    minimo = np.percentile(imagen, 1)
    maximo = np.percentile(imagen, 99)

    if maximo <= minimo:
        raise ValueError("El valor máximo debe ser mayor que el mínimo para la normalización.")

    imagen = np.clip(imagen, minimo, maximo)

    #Normalizacion
    imagen = (imagen - minimo) / (maximo - minimo) * 255

    imagen = imagen.astype(np.uint8)

    #Convertir a RGB
    imagen_rgb = np.stack([imagen] * 3, axis=-1)

    return imagen_rgb

def segmentar_disco(predictor, imagen):
    """
    Segmenta el disco de la imagen utilizando SAM.
    """
    predictor.set_image(imagen)

    # Definir un punto de referencia en el centro de la imagen
    altura, ancho = imagen.shape[:2]
    punto = np.array([[ancho // 2, altura // 2]])
    etiqueta = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords = punto,
        point_labels = etiqueta,
        multimask_output = True
    )

    mejor_indice = np.argmax(scores)
    mejor_mascara = masks[mejor_indice]

    print("Scores de SAM:", scores)

    print("Mejor score:", scores[mejor_indice])

    return mejor_mascara

def guardar_mascara(mask, output_path):
    """
    Guarda la máscara.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)

    print(f"Máscara guardada en: {output_path}")

def visualizar_mascaras(imagen, masks):

    plt.figure(figsize=(10, 10))

    plt.imshow(imagen)

    for mask in masks:

        mascara = mask["segmentation"]

        plt.contour(
            mascara,
            levels=[0.5]
        )

    plt.title(
        "Estructuras detectadas por SAM"
    )

    plt.axis("off")

    plt.show()