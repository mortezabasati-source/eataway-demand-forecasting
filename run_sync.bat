@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
echo [%date% %time%] Sync started >> sync_log.txt
"%~dp0.venv\Scripts\python.exe" auto_sync.py >> sync_log.txt 2>&1
echo [%date% %time%] Sync finished >> sync_log.txt
