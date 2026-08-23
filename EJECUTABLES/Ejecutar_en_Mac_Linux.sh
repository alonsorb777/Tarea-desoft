#!/bin/bash

# 1. Ir a la carpeta raíz del proyecto (subir un nivel desde EJECUTABLES)
cd "$(dirname "$0")/.."

echo "==================================================="
echo "  Iniciando Segmentacion de Discos Protoplanetarios"
echo "==================================================="

# 2. Crear entorno virtual en la raíz si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual de Python..."
    python3 -m venv venv
fi

# 3. Activar entorno virtual
source venv/bin/activate

# 4. Instalar dependencias
echo "Verificando e instalando librerias necesarias..."
python3 -m pip install -r requirements.txt

# 5. Lanzar Streamlit
echo "Lanzando aplicacion en el navegador..."
python3 -m streamlit run demo/demo_sam.py