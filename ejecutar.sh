#!/bin/bash
echo "==================================================="
echo "  Instalando librerias e iniciando la aplicacion..."
echo "==================================================="

pip install -r requirements.txt
python3 -m streamlit run demo/demo_sam.py