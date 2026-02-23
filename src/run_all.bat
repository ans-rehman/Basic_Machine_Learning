@echo off
setlocal enabledelayedexpansion

REM -----------------------------
REM Config
REM -----------------------------
set EXE=build\Debug\BASIC_ML.exe

REM If you build Release, change to:
REM set EXE=build\Release\BASIC_ML.exe

if not exist "%EXE%" (
  echo [ERROR] Executable not found: "%EXE%"
  echo Build your project first, then re-run this script.
  pause
  exit /b 1
)

REM Create a timestamp folder so results don’t overwrite
for /f "tokens=1-3 delims=/" %%a in ("%date%") do (
  set mm=%%a
  set dd=%%b
  set yyyy=%%c
)
for /f "tokens=1-2 delims=:" %%a in ("%time%") do (
  set hh=%%a
  set min=%%b
)
set hh=%hh: =0%
set RUNSTAMP=%yyyy%-%mm%-%dd%_%hh%-%min%

set OUTDIR=results_runs\%RUNSTAMP%
mkdir "%OUTDIR%" >nul 2>nul

REM Helper to run a dataset and copy results
call :run_one forest     forestfires
call :run_one wine-red   wine_red
call :run_one wine-white wine_white
call :run_one auto       imports85

echo.
echo [DONE] All runs finished.
echo Results saved under: %OUTDIR%
pause
exit /b 0

REM -----------------------------
REM Function: run_one
REM   %1 = CLI arg (forest/wine-red/wine-white/auto)
REM   %2 = subfolder name
REM -----------------------------
:run_one
echo.
echo ============================================
echo Running: %1
echo ============================================

REM Clean results folder so each run is isolated
if exist "results" rmdir /s /q "results"
mkdir "results" >nul 2>nul

"%EXE%" %1

REM Copy results to timestamp folder
mkdir "%OUTDIR%\%2" >nul 2>nul
xcopy /E /I /Y "results" "%OUTDIR%\%2" >nul

echo [OK] Copied results -> %OUTDIR%\%2
exit /b 0