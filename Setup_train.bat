@echo off
chcp 65001 >nul
title MoonNeuro - установка для обучения
echo === Установка зависимостей для локального обучения ===
echo.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo Видеокарта NVIDIA не найдена - ставлю PyTorch для CPU.
    pip install torch
) else (
    echo Найдена NVIDIA - ставлю PyTorch с CUDA 12.1.
    pip install torch --index-url https://download.pytorch.org/whl/cu121
)
echo.
echo Ставлю остальные зависимости...
pip install -r requirements-train.txt
echo.
echo Готово! Теперь запусти Train.bat для обучения.
pause
