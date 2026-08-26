@echo off
REM ---------------------------------------------------------------------------
REM  TAVERN: roda tudo e empacota os resultados num zip.
REM  Duplo clique, ou:  run-all.bat gemma3:4b
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set MODEL=%1
if "%MODEL%"=="" set MODEL=gemma3:4b

where python >nul 2>nul
if errorlevel 1 (
  echo [ERRO] python nao esta no PATH.
  echo        Abra o "Anaconda Prompt" e rode:  python run_all.py --model %MODEL%
  goto :fim
)

echo.
echo === TAVERN: rodando tudo com o modelo %MODEL% ===
echo.
echo Isto vai levar de 40 min a algumas horas, dependendo da GPU.
echo Pode interromper com Ctrl+C: as fusoes ficam em cache e a
echo proxima execucao continua de onde parou.
echo.
pause

python run_all.py --backbone ollama --model %MODEL%

:fim
echo.
pause
endlocal
