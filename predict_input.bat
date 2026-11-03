@echo off
title Student Performance Live Prediction
echo ========================================================
echo   Live Student Performance Predictor (User Input)
echo ========================================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat

python user_predict.py
pause
