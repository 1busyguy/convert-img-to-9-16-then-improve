@echo off

echo.
echo ==========================================
echo Step 1: Checking Python version...
echo ==========================================
python --version

echo.
pause

echo ==========================================
echo Step 2: Creating or activating virtual environment...
echo ==========================================
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Exiting.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
pause

call venv\Scripts\activate.bat

if errorlevel 1 (
    echo Failed to activate virtual environment. Exiting.
    pause
    exit /b 1
)
echo Virtual environment activated.

echo.
pause

echo ==========================================
echo Step 3: Installing Python requirements...
echo ==========================================
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install requirements. Exiting.
    pause
    exit /b 1
)
echo Requirements installed successfully.

echo.
pause

echo ==========================================
echo Step 4: Launching Streamlit app...
echo ==========================================
python -m streamlit run main.py

echo.
pause
