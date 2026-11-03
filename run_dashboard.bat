@echo off
title Student Academic Performance Predictor
echo ========================================================
echo   Student Academic Performance Prediction System
echo ========================================================
echo.
echo [1/2] Activating Python Virtual Environment...
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo [2/2] Starting Web Dashboard on http://127.0.0.1:5000 ...
echo.
echo Open your web browser and go to:
echo   http://127.0.0.1:5000
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo ========================================================
echo.

python app.py
pause
