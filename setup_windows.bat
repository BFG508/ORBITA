@echo off
echo ================================================
echo ORBITA: INITIALIZING SETUP
echo ================================================

echo.
echo [1/2] Creating the virtual environment...
py -m venv .venv
call .venv\Scripts\activate

echo.
echo [2/2] Installing dependencies...
pip install -r requirements.txt

echo.
echo ================================================
echo ALL TASKS COMPLETED SUCCESSFULLY
echo ================================================
pause