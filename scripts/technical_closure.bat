@echo off
REM =====================================================================
REM technical_closure.bat
REM
REM Regenerates every artifact referenced in the TFM (Annex B):
REM   - Global dataset and model training for all 5 architectures
REM   - K-Fold Cross-Validation for all 5 architectures
REM   - Time-domain and space-domain benchmarks for all 5 architectures
REM   - All visualization modes (single x5, ablation, metrics)
REM   - PEP-8 linting (flake8) and automated tests (pytest)
REM   - Final artifact audit
REM
REM Usage:
REM   scripts\technical_closure.bat
REM =====================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0\.."
set PYTHON=.venv\Scripts\python.exe

echo ================================================================
echo  ORBITA TECHNICAL CLOSURE
echo ================================================================

REM =====================================================================
REM 1. DATASET GENERATION
REM =====================================================================
echo [%date% %time%] START generate_global_dataset
%PYTHON% src/generate_base_dataset.py
if errorlevel 1 goto :fail
echo [%date% %time%] DONE generate_global_dataset

REM =====================================================================
REM 2. GLOBAL MODEL TRAINING (5 architectures)
REM =====================================================================
for %%A in (resnet mlp lstm linear tree) do (
    echo [%date% %time%] START train_%%A
    %PYTHON% src/train_base.py --model_type %%A
    if errorlevel 1 goto :fail
    echo [%date% %time%] DONE train_%%A
)

REM =====================================================================
REM 3. K-FOLD CROSS-VALIDATION (5 architectures)
REM =====================================================================
for %%A in (resnet mlp lstm linear tree) do (
    echo [%date% %time%] START cv_%%A
    %PYTHON% src/train_cv.py --model_type %%A --folds 5
    if errorlevel 1 goto :fail
    echo [%date% %time%] DONE cv_%%A
)

REM =====================================================================
REM 4. BENCHMARKS (5 architectures)
REM    ResNet: space-only (time-domain comes from MoE pipeline)
REM    Others: full (time-domain + space-domain)
REM =====================================================================
echo [%date% %time%] START benchmark_resnet_space
%PYTHON% src/benchmark.py --model_type resnet --mode_choice 2 --quiet --seed 42
if errorlevel 1 goto :fail
echo [%date% %time%] DONE benchmark_resnet_space

for %%A in (mlp lstm linear tree) do (
    echo [%date% %time%] START benchmark_%%A_full
    %PYTHON% src/benchmark.py --model_type %%A --mode_choice 3 --quiet --seed 42
    if errorlevel 1 goto :fail
    echo [%date% %time%] DONE benchmark_%%A_full
)

REM =====================================================================
REM 5. SIMULATE MISSION (Active Learning / GPS Reset)
REM =====================================================================
echo [%date% %time%] START simulate_mission
%PYTHON% src/simulate_mission.py
if errorlevel 1 goto :fail
echo [%date% %time%] DONE simulate_mission

REM =====================================================================
REM 6. FIGURES (all modes)
REM =====================================================================
for %%A in (resnet mlp lstm linear tree) do (
    echo [%date% %time%] START figures_%%A
    %PYTHON% src/visualize_benchmark.py --mode single --model_type %%A
    if errorlevel 1 goto :fail
    echo [%date% %time%] DONE figures_%%A
)

echo [%date% %time%] START figures_ablation
%PYTHON% src/visualize_benchmark.py --mode ablation
if errorlevel 1 goto :fail
echo [%date% %time%] DONE figures_ablation

echo [%date% %time%] START figures_metrics
%PYTHON% src/visualize_benchmark.py --mode metrics
if errorlevel 1 goto :fail
echo [%date% %time%] DONE figures_metrics

echo [%date% %time%] START figures_resnet_ablation
%PYTHON% src/plot_resnet_ablation_time.py --only-figures
if errorlevel 1 goto :fail
echo [%date% %time%] DONE figures_resnet_ablation

echo [%date% %time%] START figures_regional_ablation
%PYTHON% src/plot_expert_mesh.py
if errorlevel 1 goto :fail
echo [%date% %time%] DONE figures_regional_ablation

REM =====================================================================
REM 7. QUALITY GATES
REM =====================================================================
echo [%date% %time%] START flake8
%PYTHON% -m flake8 src tests
if errorlevel 1 goto :fail
echo [%date% %time%] DONE flake8

echo [%date% %time%] START pytest
%PYTHON% -m pytest tests/ -v
if errorlevel 1 goto :fail
echo [%date% %time%] DONE pytest

echo [%date% %time%] START audit_results
%PYTHON% src/audit_results.py
if errorlevel 1 goto :fail
echo [%date% %time%] DONE audit_results

REM =====================================================================
REM 8. CLEANUP
REM =====================================================================
for /d /r %%D in (__pycache__) do if exist "%%D" rd /s /q "%%D"
for /d /r %%D in (.pytest_cache) do if exist "%%D" rd /s /q "%%D"

echo ================================================================
echo  TECHNICAL CLOSURE COMPLETE
echo ================================================================
goto :eof

:fail
echo ================================================================
echo  TECHNICAL CLOSURE FAILED
echo ================================================================
exit /b 1
