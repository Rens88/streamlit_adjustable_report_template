@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%" || goto :cd_error

set "PYTHON="
py -3 -c "import sys" >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    python -c "import sys" >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON=python"
    ) else (
        echo Python 3 is required but was not found. Please install Python 3.x and try again.
        goto :fail
    )
)

if not exist ".venv\\Scripts\\python.exe" (
    echo Creating virtual environment in .venv...
    %PYTHON% -m venv ".venv" || goto :fail
)

call ".venv\\Scripts\\activate.bat" || goto :fail

echo Upgrading pip...
python -m pip install --upgrade pip || goto :fail

echo Installing requirements...
python -m pip install -r requirements.txt || goto :fail

echo Starting app...
python -m streamlit run app_framework.py
if %errorlevel% neq 0 goto :fail

endlocal
goto :eof

:cd_error
echo Could not change directory to the app folder.
goto :pause_fail

:fail
echo.
echo Something went wrong. Try deleting the .venv folder and double-clicking again.
:pause_fail
echo.
pause

