@echo on
cd /d "%~dp0"
call python -m streamlit run NST_with_gui.py
pause
