@echo off
setlocal

REM Run PrintStore viewer using the workspace virtual environment.
REM You can pass an optional folder path (including UNC paths) or flags like --debug.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "%~dp0app.py" %*
) else (
  echo ERROR: Virtual environment not found at "%~dp0.venv".
  echo Open a terminal in this folder and create it, then install deps.
  echo.
  echo Expected: .venv\Scripts\python.exe
  pause
  exit /b 1
)

endlocal
