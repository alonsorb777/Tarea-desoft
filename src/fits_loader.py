from astropy.io import fits
import numpy as np


def cargar_fits(archivo):
    """
    Carga una imagen FITS y devuelve sus datos en 2D.
    """

    with fits.open(archivo) as hdul:
        datos = hdul[0].data

    print("Forma original:", datos.shape)

    imagen = datos[0, 0]

    print("Forma de la imagen:", imagen.shape)
    print("Valor máximo:", np.max(imagen))
    print("Valor mínimo:", np.min(imagen))
    print("Media:", np.nanmean(imagen))
    print("Mediana:", np.nanmedian(imagen))
    print("Desviación estándar:", np.nanstd(imagen))

    return imagen