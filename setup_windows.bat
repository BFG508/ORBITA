@echo off
echo ================================================
echo ORBITA: INITIALIZING SETUP
echo ================================================

echo.
echo [1/3] Cloning the repository...
git clone https://github.com/BFG508/ORBITA.git
cd ORBITA

echo.
echo [2/3] Creating the virtual environment...
py -m venv .venv
call .venv\Scripts\activate

echo.
echo [3/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo ================================================
echo ALL TASKS COMPLETED SUCCESSFULLY
echo ================================================
pause