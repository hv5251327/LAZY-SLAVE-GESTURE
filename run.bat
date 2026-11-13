@echo off
cd /d "%~dp0"
echo Starting Hand Cursor Controller...
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py hand_cursor_controller.py
) else (
    python hand_cursor_controller.py
)
pause
