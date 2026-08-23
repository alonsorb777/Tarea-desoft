#!/bin/bash
echo "==================================================="
echo "  Iniciando Segmentacion de Discos Protoplanetarios"
echo "==================================================="

# 1. Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual de Python..."
    python3 -m venv venv
fi

# 2. Activar entorno virtual
source venv/bin/activate

# 3. Instalar dependencias
echo "Verificando e instalando librerias necesarias..."
pip install -r requirements.txt

# 4. Lanzar la app
echo "Lanzando aplicacion en el navegador..."
streamlit run demo/demo_sam.py