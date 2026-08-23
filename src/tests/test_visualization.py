import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from visualization import mostrar_imagen

def test_mostrar_imagen():
    imagen = np.random.rand(10,10)
    mostrar_imagen(imagen) 
    assert plt.gcf() is not None
    plt.close()

def test_mostrar_imagen_con_nan():
    imagen = np.array([[1, 2, np.nan],[4, 5, 6],[7, np.nan, 9]])

    mostrar_imagen(imagen)

    assert plt.gcf() is not None
    plt.close()

def test_mostrar_imagen_valores_constantes():
    imagen = np.full((10, 10),5.0)
    mostrar_imagen(imagen)

    assert plt.gcf() is not None
    plt.close()