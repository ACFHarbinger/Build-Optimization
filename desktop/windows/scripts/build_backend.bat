@echo off
REM Build the C++ backend pybind11 extension on Windows.
cd /d "%~dp0..\..\..\backend"

pixi run build
pixi run install
