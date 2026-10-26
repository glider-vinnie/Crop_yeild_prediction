@echo off
setlocal
set "PYTHON_EXE=%LocalAppData%\Python\bin\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m pip install -r requirements.txt
"%PYTHON_EXE%" -m streamlit run app.py
pause
