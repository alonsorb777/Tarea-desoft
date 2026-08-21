import argparse
from pathlib import Path

import requests

BASE_URL = "https://bulk.cv.nrao.edu/almadata/lp/DSHARP/images/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def descargar_continuum_dsharp(objetivo, carpeta_destino=None):
    """
    Descarga un archivo .fits del continuo de DSHARP y lo guarda en la carpeta data.
    """
    carpeta_destino = Path(carpeta_destino) if carpeta_destino else DEFAULT_DATA_DIR
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    nombre_archivo = f"{objetivo}_continuum.fits"
    url = f"{BASE_URL}{nombre_archivo}"
    ruta_guardado = carpeta_destino / nombre_archivo

    print(f"Iniciando descarga de {nombre_archivo}...")
    print(f"Guardando en: {ruta_guardado}")

    try:
        respuesta = requests.get(url, stream=True, timeout=30)
        respuesta.raise_for_status()

        with ruta_guardado.open("wb") as archivo:
            for chunk in respuesta.iter_content(chunk_size=8192):
                if chunk:
                    archivo.write(chunk)

        print("Descarga exitosa.\n")

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar {nombre_archivo}. Detalle: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga archivos continuum de DSHARP y los guarda en la carpeta data."
    )
    parser.add_argument(
        "objetivo",
        nargs="+",
        help="Nombre(s) del disco a descargar, por ejemplo AS209 o HTLup",
    )
    parser.add_argument(
        "--destino",
        default=str(DEFAULT_DATA_DIR),
        help="Carpeta donde guardar los archivos .fits. Por defecto: la carpeta data del proyecto.",
    )
    args = parser.parse_args()

    for disco in args.objetivo:
        descargar_continuum_dsharp(disco, args.destino)