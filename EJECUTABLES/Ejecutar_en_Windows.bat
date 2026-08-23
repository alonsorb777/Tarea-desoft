@echo off
:: Asegurar que el script se ejecute exactamente en la carpeta donde está guardado
cd /d "%~dp0"

echo ===================================================
echo   Iniciando Segmentacion de Discos Protoplanetarios
echo ===================================================

:: 1. Crear entorno virtual si no existe
if not exist "venv" (
    echo Creando entorno virtual de Python...
    python -m venv venv
)

:: 2. Activar entorno virtual
call venv\Scripts\activate.bat

:: 3. Instalar/Actualizar dependencias
echo Verificando e instalando librerias necesarias...
python -m pip install -r requirements.txt

:: 4. Lanzar la app de Streamlit usando el Python del entorno virtual
echo Lanzando aplicacion en el navegador...
python -m streamlit run demo/demo_sam.py

pause