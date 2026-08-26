@echo off
REM ---------------------------------------------------------------------------
REM  TAVERN: traz os 10 commits do bundle, DEIXA A ARVORE DE TRABALHO NELES,
REM  e (opcionalmente) publica no GitHub.
REM
REM  No PowerShell chame com  .\push-tavern.bat   (o ponto-barra e obrigatorio)
REM  No prompt do cmd, ou com duplo clique, basta  push-tavern.bat
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set BUNDLE=%~dp0TAVERN-updated.bundle
set BRANCH=tavern-thesis-framework
set WORK=%~dp0TAVERN-push

echo.
echo === TAVERN: atualizando o repositorio ===
echo.

if not exist "%BUNDLE%" (
  echo [ERRO] Nao encontrei o bundle em:
  echo        %BUNDLE%
  echo        Coloque este .bat na mesma pasta do TAVERN-updated.bundle.
  goto :fim
)

where git >nul 2>nul
if errorlevel 1 (
  echo [ERRO] git nao esta no PATH. Instale o Git for Windows e tente de novo.
  goto :fim
)

echo [1/5] Conferindo o bundle...
git bundle verify "%BUNDLE%"
if errorlevel 1 (
  echo [ERRO] Bundle invalido ou incompleto.
  goto :fim
)

if exist "%WORK%\.git" (
  echo [2/5] Reaproveitando o clone em %WORK%
  git -C "%WORK%" fetch origin
) else (
  echo [2/5] Clonando neemias8/TAVERN...
  git clone https://github.com/neemias8/TAVERN.git "%WORK%"
  if errorlevel 1 (
    echo [ERRO] Falha no clone. Verifique a rede e o acesso ao repositorio.
    goto :fim
  )
)

echo [3/5] Trazendo os commits do bundle para a branch %BRANCH%...
git -C "%WORK%" fetch --update-head-ok --force "%BUNDLE%" main:%BRANCH%
if errorlevel 1 (
  echo [ERRO] Nao consegui trazer a branch do bundle.
  goto :fim
)

REM  ESTE e o passo que faltava nas instrucoes anteriores: sem o checkout, a
REM  pasta continua com o codigo ANTIGO -- requirements.txt velho, sem
REM  run_all.py, sem run-all.bat.
echo [4/5] Trocando a arvore de trabalho para %BRANCH%...
git -C "%WORK%" checkout %BRANCH%
if errorlevel 1 (
  echo [ERRO] O checkout falhou. Se houver alteracoes locais, rode
  echo        git -C "%WORK%" stash
  echo        e tente de novo.
  goto :fim
)

echo.
echo Confirmando que a arvore esta certa:
git -C "%WORK%" rev-parse --abbrev-ref HEAD
if exist "%WORK%\run_all.py" (
  echo   ok   run_all.py presente
) else (
  echo   [ERRO] run_all.py nao apareceu. Algo deu errado no checkout.
  goto :fim
)

echo.
echo Commits novos em relacao a main do GitHub:
git -C "%WORK%" log --oneline origin/main..%BRANCH%
echo.
set /p OK="Publicar no GitHub agora? (S/N): "
if /i not "%OK%"=="S" (
  echo Nada foi enviado. A branch local esta pronta em %WORK%.
  goto :rodar
)

echo [5/5] Enviando...
git -C "%WORK%" push -u origin %BRANCH%
if errorlevel 1 (
  echo.
  echo [ERRO] O push falhou. Se pediu login, autentique e rode de novo.
  goto :fim
)
echo.
echo Publicado. Abra o Pull Request em:
echo   https://github.com/neemias8/TAVERN/compare/main...%BRANCH%
echo.
echo Se preferir ir direto para a main, no lugar do PR:
echo   git -C "%WORK%" push origin %BRANCH%:main

:rodar
echo.
echo ---------------------------------------------------------------------------
echo  PROXIMO PASSO -- rodar tudo:
echo.
echo    cd /d "%WORK%"
echo    pip install -r requirements.txt
echo    python -m spacy download en_core_web_sm
echo    ollama pull gemma3:4b
echo    python run_all.py
echo.
echo  No PowerShell, o .bat precisa de ponto-barra:  .\run-all.bat
echo  Mas  python run_all.py  funciona igual nos dois.
echo ---------------------------------------------------------------------------

:fim
echo.
pause
endlocal
