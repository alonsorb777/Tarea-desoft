import numpy as np
import pytest

from src.segmentacion.sam_segmentacion import (
    preparar_imagen,
    reducir_imagen_para_sam,
    convertir_coordenadas,
    ordenar_mascaras,
    segmentar_disco
)



# Preparar imagen


def test_preparar_imagen():

    imagen = np.random.random(
        (100, 100)
    ).astype(
        np.float32
    )

    resultado = preparar_imagen(
        imagen
    )

    assert resultado.shape == (
        100,
        100,
        3
    )

    assert resultado.dtype == np.uint8

    assert resultado.min() >= 0

    assert resultado.max() <= 255


def test_preparar_imagen_con_nan():

    imagen = np.random.random(
        (100, 100)
    ).astype(
        np.float32
    )

    imagen[10, 10] = np.nan

    resultado = preparar_imagen(
        imagen
    )

    assert resultado.shape == (
        100,
        100,
        3
    )

    assert np.isfinite(
        resultado
    ).all()


def test_preparar_imagen_constante():

    imagen = np.ones(
        (100, 100),
        dtype=np.float32
    )

    with pytest.raises(ValueError):

        preparar_imagen(
            imagen
        )



# Reducir imagen


def test_reducir_imagen():

    imagen = np.zeros(
        (2000, 3000, 3),
        dtype=np.uint8
    )

    resultado, escala = (
        reducir_imagen_para_sam(
            imagen,
            max_size=1500
        )
    )

    assert resultado.shape[0] <= 1500

    assert resultado.shape[1] <= 1500

    assert escala < 1


def test_no_reducir_imagen():

    imagen = np.zeros(
        (500, 500, 3),
        dtype=np.uint8
    )

    resultado, escala = (
        reducir_imagen_para_sam(
            imagen,
            max_size=1500
        )
    )

    assert resultado.shape == imagen.shape

    assert escala == 1.0



# Coordenadas

def test_convertir_coordenadas():

    x, y = convertir_coordenadas(
        100,
        200,
        0.5
    )

    assert x == 50

    assert y == 100


def test_convertir_coordenadas_invalida():

    with pytest.raises(ValueError):

        convertir_coordenadas(
            100,
            200,
            0
        )



# Ordenar máscaras


def test_ordenar_por_iou():

    masks = [
        {
            "predicted_iou": 0.5,
            "stability_score": 0.8,
            "area": 100
        },
        {
            "predicted_iou": 0.9,
            "stability_score": 0.7,
            "area": 200
        },
        {
            "predicted_iou": 0.7,
            "stability_score": 0.9,
            "area": 300
        }
    ]

    resultado = ordenar_mascaras(
        masks,
        "predicted_iou"
    )

    assert resultado[0]["predicted_iou"] == 0.9
    assert resultado[1]["predicted_iou"] == 0.7
    assert resultado[2]["predicted_iou"] == 0.5


def test_ordenar_por_estabilidad():

    masks = [
        {
            "predicted_iou": 0.9,
            "stability_score": 0.6,
            "area": 100
        },
        {
            "predicted_iou": 0.7,
            "stability_score": 0.95,
            "area": 200
        }
    ]

    resultado = ordenar_mascaras(
        masks,
        "stability_score"
    )

    assert resultado[0]["stability_score"] == 0.95


def test_ordenar_por_area():

    masks = [
        {
            "predicted_iou": 0.9,
            "stability_score": 0.8,
            "area": 100
        },
        {
            "predicted_iou": 0.7,
            "stability_score": 0.9,
            "area": 500
        }
    ]

    resultado = ordenar_mascaras(
        masks,
        "area"
    )

    assert resultado[0]["area"] == 500


def test_ordenar_criterio_invalido():

    with pytest.raises(ValueError):

        ordenar_mascaras(
            [],
            "criterio_inexistente"
        )



# Predictor simulado para pruebas


class PredictorFalso:

    def __init__(self):

        self.imagen = None

    def set_image(self, imagen):

        self.imagen = imagen

    def predict(
        self,
        point_coords,
        point_labels,
        multimask_output=True
    ):

        altura, ancho = (
            self.imagen.shape[:2]
        )

        mask1 = np.zeros(
            (altura, ancho),
            dtype=bool
        )

        mask2 = np.zeros(
            (altura, ancho),
            dtype=bool
        )

        mask3 = np.zeros(
            (altura, ancho),
            dtype=bool
        )

        mask1[20:40, 20:40] = True

        mask2[10:50, 10:50] = True

        mask3[30:60, 30:60] = True

        masks = np.array(
            [
                mask1,
                mask2,
                mask3
            ]
        )

        scores = np.array(
            [
                0.60,
                0.95,
                0.75
            ]
        )

        return (
            masks,
            scores,
            None
        )



# Segmentación 

def test_segmentar_disco():

    predictor = PredictorFalso()

    imagen = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    puntos = [
        [30, 30]
    ]

    etiquetas = [
        1
    ]

    resultado = segmentar_disco(
        predictor,
        imagen,
        puntos,
        etiquetas
    )

    assert "mask" in resultado
    assert "score" in resultado
    assert "all_masks" in resultado
    assert "all_scores" in resultado

    assert resultado["mask"].shape == (
        100,
        100
    )

    assert resultado["score"] == 0.95


def test_segmentar_sin_puntos():

    predictor = PredictorFalso()

    imagen = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    with pytest.raises(ValueError):

        segmentar_disco(
            predictor,
            imagen,
            [],
            []
        )


def test_segmentar_puntos_y_etiquetas_diferentes():

    predictor = PredictorFalso()

    imagen = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    puntos = [
        [30, 30],
        [50, 50]
    ]

    etiquetas = [
        1
    ]

    with pytest.raises(ValueError):

        segmentar_disco(
            predictor,
            imagen,
            puntos,
            etiquetas
        )


def test_segmentar_etiqueta_invalida():

    predictor = PredictorFalso()

    imagen = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    puntos = [
        [30, 30]
    ]

    etiquetas = [
        2
    ]

    with pytest.raises(ValueError):

        segmentar_disco(
            predictor,
            imagen,
            puntos,
            etiquetas
        )