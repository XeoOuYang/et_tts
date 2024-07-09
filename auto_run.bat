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


def start_script():
    scrip_dir = 'E:\\et_tt_live'
    # subprocess.Popen(['python', f'{scrip_dir}\\main.py'], cwd=scrip_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
    subprocess.run(['python', f'{scrip_dir}\\main.py'], cwd=scrip_dir, shell=True)

def auto_run():
    print('==' * 48)
    # 检查并启动api服务
    if not check_api():
        task_api = threading.Thread(target=startup_api)
        task_api.start()
        time.sleep(1)
    # 检查api服务
    while not check_api():
        time.sleep(1)
    # 启动直播脚本
    task_script = threading.Thread(target=start_script)
    task_script.start()
    task_script.join()
    print('==' * 48)

if __name__ == '__main__':
    auto_run()