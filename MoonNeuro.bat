@echo off
chcp 65001 >nul
title MoonNeuro
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama не установлена. Скачай: https://ollama.com/download
    pause
    exit /b
)
ollama list | findstr /i "moonneuro" >nul
if errorlevel 1 (
    echo Создаю модель MoonNeuro из GGUF...
    ollama create moonneuro -f Modelfile
)
ollama run moonneuro
pause
