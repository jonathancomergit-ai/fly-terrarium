@echo off
rem Fly Terrarium launcher: starts the GPU brain server and opens the page.
cd /d "%~dp0"
if not exist data\brain.npz (
  echo First run: building data\brain.npz from data\raw ^(about a minute^)...
  python prep.py || pause
)
if not exist data\eyes.npz (
  python prep_eyes.py
)
rem the eye environment (Python 3.12 + flyvis) runs everything if it exists; plain python = no Eyes page
if exist .venv-eye\Scripts\python.exe (
  .venv-eye\Scripts\python.exe server.py
) else (
  python server.py
)
rem only keep the window open if something went wrong
if errorlevel 1 pause
