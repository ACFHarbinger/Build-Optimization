@echo off
REM Bootstrap the Build-Optimization dev environment on Windows.
cd /d "%~dp0..\..\.."

where uv >nul 2>nul || (powershell -c "irm https://astral.sh/uv/install.ps1 | iex")
where pixi >nul 2>nul || (powershell -c "irm https://pixi.sh/install.ps1 | iex")

uv sync
pushd backend
pixi install
popd
npm install --workspaces

echo Setup complete. Activate with: .venv\Scripts\activate.bat
