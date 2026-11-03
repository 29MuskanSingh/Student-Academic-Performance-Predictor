@echo off
title Student Performance ML Analysis (Terminal & Graphs)
echo ========================================================
echo   Running Complete Student Performance ML Pipeline
echo ========================================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat

python main.py
pause
