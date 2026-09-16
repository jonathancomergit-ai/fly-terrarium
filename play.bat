@echo off
rem Fly Terrarium launcher: starts the GPU brain server and opens the page.
cd /d "%~dp0"
if not exist data\brain.npz (
  echo First run: building data\brain.npz from data\raw ^(about a minute^)...
  python prep.py || pause
)
python server.py
rem only keep the window open if something went wrong
if errorlevel 1 pause
