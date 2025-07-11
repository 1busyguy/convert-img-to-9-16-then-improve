@echo off
REM --- Check Python version ---
python --version 2>nul | findstr /r "^Python 3\.[89].*" >nul
if %errorlevel% neq 0 (
    echo Python 3.8 or newer is required. Please install the latest Python 3.x and add it to PATH.
    pause
    exit /b
)

REM --- Create venv if it does not exist ---
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM --- Activate venv ---
call venv\Scripts\activate.bat
echo Virtual environment activated.

REM --- Always upgrade pip and install requirements ---
pip install --upgrade streamlit
pip install --upgrade pip
pip install -r requirements.txt

REM --- Launch Streamlit app ---
python streamlit run img_test.py

REM --- When finished, deactivate env ---
deactivate
