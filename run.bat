@echo off
echo =========================================
echo   AC-Net Liver Tumor Segmentation
echo =========================================
call venv\Scripts\activate
echo Starting Flask server...
echo Open http://localhost:5000 in your browser
python app.py
pause
