@echo off
REM Launch the Tauri Studio in dev mode on Windows.
cd /d "%~dp0..\..\..\frontend"

npm run tauri:dev
