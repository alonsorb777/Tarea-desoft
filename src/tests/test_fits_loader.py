import numpy as np
from astropy.io import fits

from fits_loader import cargar_fits

def test_cargar_fits():
    datos = np.random.rand(1, 1, 10, 10)
    archivo = "test_image.fits"
    fits.writeto(archivo, datos, overwrite=True)

    imagen = cargar_fits(archivo)

    assert imagen is not None
    assert imagen.shape == (10,10)