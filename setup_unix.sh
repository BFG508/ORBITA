#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "================================================"
echo "ORBITA: INITIALIZING SETUP"
echo "================================================"

echo -e "\n[1/3] Cloning the repository..."
git clone https://github.com/BFG508/ORBITA.git
cd ORBITA

echo -e "\n[2/3] Creating the virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo -e "\n[3/3] Installing dependencies..."
pip install -r requirements.txt

echo -e "\n================================================"
echo "ALL TASKS COMPLETED SUCCESSFULLY"
echo "================================================"