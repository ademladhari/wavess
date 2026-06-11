@echo off
REM Restore .venv\Scripts\python.exe when it becomes 0 bytes (Defender quarantine).
set "TEMPLATE=%LocalAppData%\Programs\Python\Python310\Lib\venv\scripts\nt\python.exe"
set "TARGET=%~dp0.venv\Scripts\python.exe"

if not exist "%TEMPLATE%" (
    echo Template not found: %TEMPLATE%
    echo Edit this script if your Python is installed elsewhere.
    exit /b 1
)

copy /Y "%TEMPLATE%" "%TARGET%" >nul
for %%A in ("%TARGET%") do set SIZE=%%~zA
echo Restored python.exe size=%SIZE% bytes

"%TARGET%" --version
if errorlevel 1 (
    echo python.exe still fails — add Defender exclusion for %~dp0.venv
    exit /b 1
)

echo OK. Add Windows Defender exclusion for: %~dp0.venv
