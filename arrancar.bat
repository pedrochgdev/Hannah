@echo off
title Hannah AI - Sistema

if "%1"=="stop" (
    echo Deteniendo Hannah...
    cd /d "%~dp0"
    docker compose down
    echo Sistema detenido.
    pause
    exit /b 0
)

echo.
echo ==========================================
echo   Hannah AI - Arranque
echo ==========================================
echo.

echo Verificando Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker no esta corriendo.
    echo        Abre Docker Desktop y vuelve a intentarlo.
    pause
    exit /b 1
)
echo OK - Docker disponible

echo.
echo Levantando servicios (rebuild automatico si hay cambios)...
cd /d "%~dp0"
docker compose up -d --build
if errorlevel 1 (
    echo ERROR: Fallo docker compose up.
    echo        Revisa el error arriba.
    pause
    exit /b 1
)

echo.
echo Esperando que la API este lista...
set intentos=0
:esperar
set /a intentos+=1
if %intentos% gtr 30 goto timeout_api
curl -sf http://localhost:8000/api/v1/health >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto esperar
)
echo OK - API lista
goto mostrar

:timeout_api
echo La API tarda mas de lo normal.
echo Revisa con: docker compose logs hannah-api

:mostrar
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (
    set MYIP=%%a
    goto ipok
)
set MYIP= localhost
:ipok
set MYIP=%MYIP: =%

echo ==========================================
echo   SISTEMA LISTO
echo ==========================================
echo.
echo   URL para los usuarios:
echo   http://%MYIP%:8000
echo.
echo   Credenciales:
echo     luis   / hannah1  (admin)
echo     user2  / hannah2
echo     user3  / hannah3
echo     user4  / hannah4
echo     user5  / hannah5
echo.
echo   Para detener: arrancar.bat stop
echo ==========================================
echo.

set /p ABRIR=Abrir navegador ahora? (s/n): 
if /i "%ABRIR%"=="s" start http://localhost:8000

pause
