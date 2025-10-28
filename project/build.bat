@echo on
cd /d %~dp0
set PY=python

rem 
%PY% -m pip show pyinstaller >nul 2>&1 || %PY% -m pip install -U pyinstaller

rem 
%PY% -m PyInstaller ^
  --noconfirm --clean ^
  --name KZTRUB_GUI ^
  --noconsole ^
  --add-data "app;app" ^
  --add-data "src;src" ^
  --hidden-import matplotlib.backends.backend_qt5agg ^
  main.py

echo ==== DONE ====
dir dist\KZTRUB_GUI
pause
