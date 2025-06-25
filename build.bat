@echo off
REM ===========================================
REM Build and Run Any C++ File in This Project
REM Usage: build.bat relative_path_to_cpp_file
REM Example: build.bat examples\single_neuron_example.cpp
REM ===========================================

if "%1"=="" (
    echo [ERROR] No input file provided.
    echo Usage: build.bat ^<relative_path_to_cpp_file^>
    exit /b 1
)

REM Compiler and flags
set COMPILER=g++
set CFLAGS=-std=c++17 -Iinclude

REM Output file
set OUTPUT=build\main.exe

REM Create build directory if not exists
if not exist build mkdir build

REM Compile everything plus user-specified file
%COMPILER% %CFLAGS% ^
%1 ^
src\Layers\Activation_utils.cpp ^
src\Layers\ActivationLayer.cpp ^
src\Layers\DenseLayer.cpp ^
src\Metrics\Correlations.cpp ^
src\Metrics\Losses\1_mse.cpp ^
src\Metrics\Losses\2_mae.cpp ^
src\Metrics\Losses\3_bce.cpp ^
src\Metrics\Losses\4_cross_entropy.cpp ^
src\Metrics\Losses\5_hinge.cpp ^
src\Models\Sequential.cpp ^
src\Optimizers\BaseOptim.cpp ^
src\Optimizers\SGD.cpp ^
src\Optimizers\BatchGD.cpp ^
src\Optimizers\MiniBatchGD.cpp ^
src\Preprocessing\Dataset_utils.cpp ^
src\Preprocessing\Preprocessing.cpp ^
src\Utils\Initialization.cpp ^
-o %OUTPUT%

REM Build status
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b %errorlevel%
)

echo [INFO] ===== Build Successful =====
echo Running %OUTPUT%...
%OUTPUT%

pause
