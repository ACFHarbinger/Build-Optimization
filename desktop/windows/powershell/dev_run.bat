@echo off
REM Launch the Tauri Studio in dev mode on Windows.
cd /d "%~dp0..\..\..\app"

npm run tauri:dev
