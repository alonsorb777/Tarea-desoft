@echo off
echo ===================================================
echo   Iniciando Segmentacion de Discos Protoplanetarios
echo ===================================================

::1. Crear entorno virtual si no existe
if not exist "venv" (
    echo Creando entorno virtual de Python...
    python -m venv venv
)

:: 2. Activar entorno virtual
call venv\Scripts\activate.bat

:: 3. Instalar/Actualizar dependencias
echo Verificando e instalando librerias necesarias...
pip install -r requirements.txt

:: 4. Lanzar la app de Streamlit
echo Lanzando aplicacion en el navegador...
streamlit run demo/demo_sam.py

pause