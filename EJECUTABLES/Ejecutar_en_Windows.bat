@echo off
TITLE Lanzador de Aplicacion - Segmentacion SAM

:: 1. Ir a la carpeta raiz del proyecto
cd /d "%~dp0.."

echo ====================================================
echo  Iniciando Segmentacion de Discos Protoplanetarios
echo ====================================================

:: 2. Instalar Visual C++ Redistributable silenciosamente
echo.
echo [1/5] Verificando dependencias de Windows (C++ Runtime)...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando dependencias necesarias de C++...
    curl -s -L "https://aka.ms/vs/17/release/vc_redist.x64.exe" -o "%temp%\vc_redist.x64.exe"
    start /wait "%temp%\vc_redist.x64.exe" /quiet /norestart
    del "%temp%\vc_redist.x64.exe"
)

:: 3. Verificar e instalar Git si no existe
echo.
echo [2/5] Verificando Git...
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo Git no detectado. Instalando Git automaticamente...
    winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
)

:: 4. Verificar e instalar Python 3.10 descargando el instalador oficial
echo.
echo [3/5] Verificando Python...
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Python no detectado. Descargando e instalando Python 3.10 de forma silenciosa...
        curl -s -L "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -o "%temp%\python_installer.exe"
        start /wait "%temp%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        del "%temp%\python_installer.exe"
    )
)

:: Forzar rutas de Python en el PATH de esta sesión de consola por si no se han refrescado
set "PATH=%ProgramFiles%\Python310;%ProgramFiles%\Python310\Scripts;%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;%PATH%"

:: 5. Crear entorno virtual si no existe
echo.
echo [4/5] Configurando entorno virtual...
if not exist "venv" (
    echo Creando entorno virtual de Python...
    py -3.10 -m venv venv 2>nul || python -m venv venv
)

:: 6. Activar entorno virtual e instalar paquetes
call venv\Scripts\activate.bat

echo.
echo [5/5] Verificando e instalando librerias de Python...
python -m pip install --upgrade pip

:: Fuerza instalación de PyTorch CPU para máxima compatibilidad
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

:: 7. Lanzar Streamlit
echo.
echo Lanzando aplicacion en el navegador...
python -m streamlit run demo/demo_sam.py

pause