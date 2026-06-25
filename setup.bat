@echo off
echo =========================================
echo   AC-Net Liver Tumor Segmentation Setup
echo =========================================

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install Flask==3.0.0 Werkzeug==3.0.1 Pillow numpy

echo.
echo Installing PyTorch (CPU version - works on all machines)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo ✅ Setup complete!
echo Run 'run.bat' to start the server.
pause
