1>2# : ^
'''
@echo off
call conda activate et_tts_env
python %~f0 %*
exit /b
rem ^
'''
import subprocess
import threading

import requests
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_id', type=str, default='okL6Yo', help='当前直播脚本配置')
parser.add_argument('--run_mode', type=str, default='1', help='运行模式(0表示直播，1表示生成，2表示互动)')
parser.add_argument('--obs_video_port', type=str, default='0', help='obs视频流端口')
parser.add_argument('--obs_audio_port', type=str, default='0', help='obs音频流端口')
args = parser.parse_args()

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


def start_script(config_id, run_mode=1, obs_video_port=0, obs_audio_port=0):
    scrip_dir = 'E:\\et_tt_live'
    subprocess.run(['python', f'{scrip_dir}\\main.py', '--config_id', config_id, '--run_mode', str(run_mode), '--obs_video_port', str(obs_video_port), '--obs_audio_port', str(obs_audio_port)], cwd=scrip_dir, shell=True)

def auto_run():
    API_RUNNING = False
    # 检查并启动api服务
    if not check_api():
        task_api = threading.Thread(target=startup_api)
        task_api.start()
        time.sleep(1)
    else:
        API_RUNNING = True
        print('API service is running...')
    # 检查api服务
    while not API_RUNNING and not check_api():
        time.sleep(1)
    API_RUNNING = True
    print('API service is startup...')
    # 启动生成脚本
    if args.run_mode == '1':
        task_script = threading.Thread(target=start_script, kwargs={'config_id':args.config_id, 'run_mode':1, 'obs_video_port':0, 'obs_audio_port':0})
        task_script.start()
        task_script.join()
    print('Script is generated...')
    # 启动直播互动
    task_script = threading.Thread(target=start_script, kwargs={'config_id':args.config_id, 'run_mode':2, 'obs_video_port':0, 'obs_audio_port':0})
    task_script.start()
    task_script.join()
    print('Interact has all done...')

if __name__ == '__main__':
    auto_run()
