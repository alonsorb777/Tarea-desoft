import numpy as np
import matplotlib.pyplot as plt
from visualization import mostrar_imagen

def test_mostrar_imagen():
    imagen = np.random.rand(10,10)
    mostrar_imagen(imagen) 
    assert plt.gcf() is not None
    plt.close()