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
echo 3. Lancer les DEUX avec Python local
echo 4. Lancer les DEUX avec Docker (recommande)
echo 5. Ouvrir la documentation API dans Chrome
echo 6. Quitter
echo.

set /p choice="Entre ton choix (1-6): "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto fastapi
if "%choice%"=="3" goto both
if "%choice%"=="4" goto docker
if "%choice%"=="5" goto docs
if "%choice%"=="6" goto end

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

:docker
echo.
echo Construction et lancement de Streamlit et FastAPI avec Docker...
docker compose up --build -d
if errorlevel 1 (
    echo Echec du lancement Docker. Verifie que Docker Desktop est demarre.
    pause
    goto end
)
echo.
echo Interface: http://localhost:8501
echo API: http://localhost:8000
echo Documentation: http://localhost:8000/docs
goto end

:docs
echo.
set "CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_PATH%" (
    echo Google Chrome est introuvable: %CHROME_PATH%
    echo Ouvre manuellement http://localhost:8000/docs dans Chrome.
    pause
    goto end
)
echo Ouverture de la documentation API dans Google Chrome...
start "" "%CHROME_PATH%" http://localhost:8000/docs
goto end

:end
echo.
echo Au revoir!
pause
