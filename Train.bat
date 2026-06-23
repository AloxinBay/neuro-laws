@echo off
chcp 65001 >nul
title MoonNeuro - обучение
echo === Локальное обучение MoonNeuro ===
echo Это займёт время (на CPU - часы, на GPU - минуты).
echo.
python scripts\train.py %*
pause
