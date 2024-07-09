1>2# : ^
'''
@echo off
call conda activate et_tts_env
python %~f0"
exit /b
rem ^
'''
import subprocess
import threading

import requests
import time

def check_api():
    url = 'http://127.0.0.1:9394/llm/tts/ver'
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.status_code == 200
    except Exception as err:
        print(err)
        return False

def startup_api():
    api_dir = 'E:\\ET_TTS'
    subprocess.Popen([f'{api_dir}\\run_daemon.bat'], cwd=api_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)


def start_script(run_mode=1):
    scrip_dir = 'E:\\et_tt_live'
    # subprocess.Popen(['python', f'{scrip_dir}\\main.py', '--run_mode', f'{run_mode}'], cwd=scrip_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    subprocess.run(['python', f'{scrip_dir}\\main.py', '--run_mode', f'{run_mode}'], cwd=scrip_dir, shell=True)

def auto_run():
    # 检查并启动api服务
    if not check_api():
        task_api = threading.Thread(target=startup_api)
        task_api.start()
        time.sleep(1)
    # 检查api服务
    while not check_api():
        time.sleep(1)
    # 启动生成脚本
    task_script = threading.Thread(target=start_script, kwargs={'run_mode':1})
    task_script.start()
    task_script.join()
    # 启动直播互动
    task_script = threading.Thread(target=start_script, kwargs={'run_mode': 2})
    task_script.start()
    task_script.join()

if __name__ == '__main__':
    auto_run()
