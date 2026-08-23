@echo off
:: 1. Ir a la carpeta raíz del proyecto (subir un nivel desde EJECUTABLES)
cd /d "%~dp0.."

echo ===================================================
echo   Iniciando Segmentacion de Discos Protoplanetarios
echo ===================================================

:: 2. Crear entorno virtual en la raíz si no existe
if not exist "venv" (
    echo Creando entorno virtual de Python...
    python -m venv venv
)

:: 3. Activar entorno virtual
call venv\Scripts\activate.bat

:: 4. Instalar dependencias desde la raíz
echo Verificando e instalando librerias necesarias...
python -m pip install -r requirements.txt

:: 5. Lanzar Streamlit apuntando a la ruta de la demo
echo Lanzando aplicacion en el navegador...
python -m streamlit run demo/demo_sam.py

pause