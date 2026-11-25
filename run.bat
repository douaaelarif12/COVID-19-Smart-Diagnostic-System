@echo off
REM Lance le prototype local de classification thoracique.

echo ================================================
echo    CLASSIFICATION THORACIQUE - LANCEMENT
echo ================================================
echo.

echo Choisis une option:
echo.
echo 1. Lancer Streamlit seulement (Frontend)
echo 2. Lancer FastAPI seulement (Backend API)
echo 3. Lancer les DEUX (Streamlit + FastAPI)
echo 4. Ouvrir la documentation API
echo 5. Quitter
echo.

set /p choice="Entre ton choix (1-5): "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto fastapi
if "%choice%"=="3" goto both
if "%choice%"=="4" goto docs
if "%choice%"=="5" goto end

echo Choix invalide!
pause
exit

:streamlit
echo.
echo Lancement de Streamlit...
echo Interface accessible sur: http://localhost:8501
echo.
streamlit run app.py
goto end

:fastapi
echo.
echo Lancement de FastAPI...
echo API accessible sur: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo.
python -m uvicorn api:app --reload
goto end

:both
echo.
echo Lancement de FastAPI et Streamlit...
echo.
echo Terminal 1: API (http://localhost:8000)
start cmd /k "python -m uvicorn api:app --reload"
timeout /t 3 >nul
echo.
echo Terminal 2: Streamlit (http://localhost:8501)
start cmd /k "streamlit run app.py"
echo.
echo Les deux services sont lances!
echo Ferme les fenetres pour arreter.
goto end

:docs
echo.
echo Ouverture de la documentation API...
start http://localhost:8000/docs
goto end

:end
echo.
echo Au revoir!
pause
