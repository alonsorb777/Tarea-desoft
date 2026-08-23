#!/bin/bash
echo "==================================================="
echo "  Instalando librerias e iniciando la aplicacion..."
echo "==================================================="

cd ..
pip install -r requirements.txt
python3 -m streamlit run demo/demo_sam.py