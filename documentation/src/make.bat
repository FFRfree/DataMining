@echo off
cd %~dp0
set LSHORT=a
set PDF=%LSHORT%.pdf
set TEMP=%LSHORT%.xdv %LSHORT%.aux %LSHORT%.log %LSHORT%.idx %LSHORT%.ind %LSHORT%.ilg %LSHORT%.out %LSHORT%.toc %LSHORT%.los %LSHORT%-example.aux %LSHORT%.synctex.gz

if "%1"=="clean" goto clean
if "%1"=="distclean" goto distclean

set TEX=xelatex
set NOPDFMODE=-interaction=nonstopmode -synctex=1 --no-pdf
set MODE=-interaction=nonstopmode -synctex=1
set MAKEINDEX=makeindex


@REM if exist %PDF% (
@REM copy %PDF% ..
@REM )
@REM exit /B

:clean
del %TEMP%
del %PDF%
exit /B

:distclean
del %TEMP%
del %PDF%
del ..\%PDF%
exit /B