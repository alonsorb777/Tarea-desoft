from astropy.io import fits
import matplotlib.pyplot as plt

archivo = "data/AS209_continuum.fits"

with fits.open(archivo) as hdul:
    datos = hdul[0].data

print("Forma original:", datos.shape)

#imagen en 2d
imagen = datos[0, 0]

print("Forma de la imagen:", imagen.shape)

plt.imshow(imagen, origin="lower")
plt.colorbar(label="Intensidad")
plt.title("AS 209 - Imagen de continuo DSHARP")
plt.xlabel("Píxel")
plt.ylabel("Píxel")
plt.show()