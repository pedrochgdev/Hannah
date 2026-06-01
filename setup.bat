@echo off
title Hannah AI - Setup

echo.
echo ==========================================
echo   Hannah AI - Setup (solo la primera vez)
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
echo Construyendo imagenes Docker...
echo (Esto puede tardar 5-10 min la primera vez)
echo.
cd /d "%~dp0"
docker compose build
if errorlevel 1 (
    echo ERROR: Fallo docker compose build.
    echo Asegurate de que Docker Desktop este abierto.
    pause
    exit /b 1
)
echo OK - Imagenes Docker listas

if not exist "%~dp0backend-hannah\.env" (
    (
        echo REDIS_URL=redis://redis:6379
        echo JWT_SECRET=hannah-local-secret-2026
        echo SESSION_TTL_SECONDS=1800
        echo FAST_MODEL_URL=http://host.docker.internal:8001/generate
        echo SLOW_MODEL_URL=http://host.docker.internal:8003/generate
    ) > "%~dp0backend-hannah\.env"
    echo OK - .env backend creado
)

if not exist "%~dp0..\hannah-backend\.env" (
    (
        echo REDIS_URL=redis://redis:6379
        echo JWT_SECRET=hannah-local-secret-2026
        echo SESSION_TTL_MINUTES=30
        echo LLM_API_KEY=placeholder
        echo CORS_ORIGIN=*
    ) > "%~dp0..\hannah-backend\.env"
    echo OK - .env gateway creado
)

echo.
echo ==========================================
echo   Setup completo.
echo   Ahora haz doble click en arrancar.bat
echo ==========================================
echo.
pause
