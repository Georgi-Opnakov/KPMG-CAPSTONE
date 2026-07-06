@echo off
setlocal

cd /d "%~dp0"
echo Starting Airbnb Intelligent Advisor...
echo.
echo The app should open in your browser at http://localhost:8501
echo Keep this window open while using the chatbot.
echo.

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0run_chatbot.ps1"
