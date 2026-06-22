@echo off
chcp 65001 >nul
title MoonNeuro
python scripts\inference.py --model outputs\merged
pause
