import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagen(imagen):
    """
    Muestra una imagen en 2d del disco protoplanetario utilizando precentiles para mejorar el contraste.
    """
    p1 = np.nanpercentile(imagen, 1)
    p99 = np.nanpercentile(imagen, 99)

    print("Percentil 1%:", p1)
    print("Percentil 99%:", p99)

    plt.imshow(
    imagen,
    origin="lower",
    cmap="inferno",
    vmin=p1,
    vmax=p99
    )

    plt.colorbar(label="Intensidad")
    plt.title("Imagen de continuo de AS 209")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")

    plt.show()