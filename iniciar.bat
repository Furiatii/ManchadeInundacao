@echo off
echo ============================================
echo   Mancha de Inundacao v2
echo   Modelagem de Ruptura de Barragens
echo ============================================
echo.
echo Verificando dependencias...
py -m pip install -r "%~dp0requirements.txt" --quiet
echo.
echo Abrindo no navegador...
start http://localhost:8501
py -m streamlit run "%~dp0app.py" --server.headless true --browser.gatherUsageStats false
pause
