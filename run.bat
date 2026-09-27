@echo off
chcp 65001 >nul
title 桥梁智巡 - 面向铁路桥梁的同视场无人机病害巡检分析系统
cd /d "%~dp0"

echo ============================================================
echo   桥梁智巡  正在启动...
echo ============================================================
echo.

rem 优先用 py 启动器；没有则退回 python
where py >nul 2>nul
if %errorlevel%==0 (
    py main.py
) else (
    python main.py
)

if errorlevel 1 (
    echo.
    echo [启动失败] 请先安装依赖：
    echo     py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
)
