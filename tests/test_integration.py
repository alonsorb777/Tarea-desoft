import numpy as np 
import pytest 
from pathlib import Path 

from src.fits_loader import cargar_fits 
from src.segmentacion.sam_segmentacion import (
    preparar_imagen,
    reducir_imagen_para_sam,
)

@pytest.mark.integration
def test_full_processing_pipeline(tmp_path):
    """Prueba de integracion:
     
    carga archivos FITS, normalizacion y preparacion RGB, redimensionamiento,
    guardado.
    """
    fits_path = Path("test_image.fits")
    assert fits_path.exists(), "El archivo de prueba test_image.fits no existe."

    data = cargar_fits(str(fits_path))
    assert data is not None, "Error al cargar la imagen FITS."
    assert isinstance(data, np.ndarray)

    rgb_image = preparar_imagen(data)
    assert rgb_image.shape[-1] == 3, "la imagen procesada debe ser RGB (3 canales)."
    assert rgb_image.dtype == np.uint8, "el tipo de dato debe ser uint8"

    imagen_reducida, escala = reducir_imagen_para_sam(rgb_image, max_size=500)
    assert imagen_reducida is not None
    assert escala > 0 

    output_file = tmp_path / "processed_image.npy"
    np.save(output_file, imagen_reducida)

    assert output_file.exists(), "El archivo de salida .npy no se guardo."
    loaded_data = np.load(output_file)
    assert np.array_equal(imagen_reducida, loaded_data)

