@echo off
echo "Starting fastapi..."
set _enable_reload_=False
if "%1"=="True" (set _enable_reload_=True)
:run_loop
REM 激活conda环境
call conda activate et_tts_env
REM 执行Python脚本
python et_fastapi.py %_enable_reload_%
REM 退出conda环境
call conda deactivate
echo "Restart fastapi..."
goto run_loop
