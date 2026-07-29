@echo off
REM Thin CLI wrapper: run an optimization from a terminal.
REM Usage: optimize.bat game=rpg policy=policy_sa
cd /d "%~dp0..\..\.."

uv run python main.py %*
