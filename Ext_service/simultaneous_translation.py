# -*- coding:utf-8 -*-
import _thread as thread
import base64
import datetime
import hashlib
import hmac
import os
import ssl
import time
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import json
from tool import pcm2wav

'''
 1、同声传译接口，可以将音频流实时翻译为不同语种的文本，并输对应的音频内容，广泛应用于国际论坛、智能会议、智慧教育、跨国交流等场景。
 '''

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class Ws_Param(object):
    # 初始化
    encoding = 'raw'

    def __init__(self, AudioFile):
        # 控制台鉴权信息
        self.APPID = APPID
        self.APISecret = APISecret
        self.APIKey = APIKey
        self.Host = "ws-api.xf-yun.com"
        self.HttpProto = "HTTP/1.1"
        self.HttpMethod = "GET"
        self.RequestUri = "/v1/private/simult_interpretation"
        self.Algorithm = "hmac-sha256"
        self.url = "ws://" + self.Host + self.RequestUri
        # self.encoding = 'raw'

        # 设置测试音频文件
        self.AudioFile = audio_path

    # 生成url
    def create_url(self):
        url = self.url
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + self.Host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.RequestUri + " HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.Host,
            "serviceId": "simult_interpretation"
        }
        url = url + '?' + urlencode(v)
        return url

    @staticmethod
    def create_params(appid, status, audio):
        param = {
            "header": {
                "app_id": appid,
                "status": status,
            },
            "parameter": {
                "ist": {
                    "accent": "mandarin",
                    "domain": "ist_ed_open",
                    "language": "zh_cn",
                    "vto": 15000,
                    "eos": 150000
                },
                "streamtrans": {
                    "from": "en",
                    "to": "cn"
                },
                "tts": {
                    "vcn": "x2_catherine",
                    "tts_results": {
                        "encoding": "raw",
                        "sample_rate": 16000,
                        "channels": 1,
                        "bit_depth": 16,
                        "frame_size": 0
                    }
                }
            },
            "payload": {
                "data": {
                    "audio": str(base64.b64encode(audio).decode('utf-8')),
                    "encoding": "raw",
                    "sample_rate": 16000,
                    "seq": 1,
                    "status": status
                }
            }
        }
        return param


    # 收到websocket消息的处理
    def on_message(self,ws, message):
        try:
            print(f"收到的消息：{message}")
        except Exception as e:
            print("receive msg,but parse exception:", e)

        # 对结果进行解析
        message = json.loads(message)
        status = message["header"]["status"]
        sid = message["header"]["sid"]
        # 接收到的识别结果写到文本
        if "recognition_results" in message['payload']:
            result = message['payload']['recognition_results']['text']
            asrresult = base64.b64decode(result).decode()
            file1 = open('output\\text\\asr.txt', "a", encoding="UTF-8")
            file1.write(asrresult)
            file1.close()

        # 接收到的翻译结果写到文本
        if "streamtrans_results" in message['payload']:
            result = message['payload']['streamtrans_results']['text']
            transresult = base64.b64decode(result).decode()
            file1 = open('output\\text\\trans.txt', "a", encoding="UTF-8")
            file1.write(transresult)
            file1.close()

        # 把接收到的音频流合成PCM
        if "tts_results" in message['payload']:
            audio = message['payload']['tts_results']['audio']
            audio = base64.b64decode(audio)
            with open('output\\audio\\trans.pcm', 'ab') as f:
                f.write(audio)

        if status == 2:
            print("session end ")
            print("本次请求的sid==》 " + sid)
            print("数据处理完毕，等待实时转译结束！同传后的音频文件请到output/audio/目录查看...")
            time.sleep(1)
            ws.close()


    # 收到websocket错误的处理
    def on_error(self, ws, error):
        print("### error:", error)


    # 收到websocket关闭的处理
    def on_close(self, ws, code, msg):
        print("### closed ###", code, msg)


    # 收到websocket连接建立的处理
    def on_open(self,ws):
        def run(*args):
            frameSize = 1280  # 每一帧的音频大小
            intervel = 0.04  # 发送音频间隔(单位:s)
            status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧
            with open(self.AudioFile, "rb") as fp:
                while True:
                    buf = fp.read(frameSize)
                    # 文件结束
                    if not buf:
                        status = STATUS_LAST_FRAME
                    # 第一帧处理
                    # 发送第一帧音频，带business 参数
                    # appid 必须带上，只需第一帧发送
                    if status == STATUS_FIRST_FRAME:
                        # print(wsParam.create_params(ws.appid, status, buf))
                        ws.send(json.dumps(self.create_params(ws.appid, status, buf)))
                        print('第一帧已发送...')
                        status = STATUS_CONTINUE_FRAME
                    # 中间帧处理
                    elif status == STATUS_CONTINUE_FRAME:
                        ws.send(json.dumps(self.create_params(ws.appid, status, buf)))
                        # print('中间帧已发送...')
                    # 最后一帧处理
                    elif status == STATUS_LAST_FRAME:
                        print('最后一帧已发送...')
                        ws.send(json.dumps(self.create_params(ws.appid, status, buf)))
                        break

                    # 模拟音频采样间隔
                    time.sleep(intervel)
            # ws.close()

        thread.start_new_thread(run, ())

    def get_audio_text(self):
        websocket.enableTrace(False)
        wsUrl = self.create_url()
        # 创建转写、转译的text文本和传译的音频
        open('output\\text\\asr.txt', 'w')
        open('output\\text\\trans.txt', 'w')
        open('output\\audio\\trans.pcm', 'w')
        ws = websocket.WebSocketApp(wsUrl, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        ws.on_open = self.on_open
        ws.appid = self.APPID
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        # 输出wav格式音频
        pcm2wav.pcm_2_wav("output\\audio\\trans.pcm", "output\\audio\\trans.wav")

if __name__ == "__main__":
    # 在控制台获取appid等信息
    APPID = '29ffdb04'
    APISecret = 'ZGJmMzEzYTQxMTA1ODBhYTNjNjQ5Y2Vk'
    APIKey = 'd03b067308ceadd521cd290d5a7b529b'
    # 音频路径
    from et_dirs import resources
    audio_path = source = os.path.join(resources, 'man_role0_ref.pcm')
    from et_base import timer
    with timer('simult_interpretation'):
        demo = Ws_Param(audio_path)
        demo.get_audio_text()



