from fits_loader import cargar_fits
from visualization import mostrar_imagen


archivo = "data/AS209_continuum.fits"

imagen = cargar_fits(archivo)

mostrar_imagen(imagen)