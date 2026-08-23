@echo off
echo ===================================================
echo   Instalando librerias e iniciando la aplicacion
echo ===================================================

cd ..
pip install -r requirements.txt
python -m streamlit run demo/demo_sam.py

pause