import requests
import uuid
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json

HOST = 'http://127.0.0.1:9394'
_ET_UUID_ = uuid.uuid1().hex


async def post_retry(url, headers, data):
    retry_times = 3  # 设置重试次数
    retry_backoff_factor = 2.0  # 设置重试间隔时间
    status_force_list = [400, 401, 403, 404, 500, 502, 503, 504]
    session = requests.Session()
    retry = Retry(total=retry_times, backoff_factor=retry_backoff_factor, status_forcelist=status_force_list)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session.post(url, headers=headers, json=data)


async def ver_async():
    url = f"{HOST}/llm/tts/ver"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    return response.status_code == 200


async def llm_async(query, role_play, context, inst_text, max_num_sentence, repetition_penalty=1.05,
                    et_uuid=_ET_UUID_, language="english"):
    url = f"{HOST}/llm/tr"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "et_uuid": et_uuid,
        "language": language,
        "use_history": False,
        "query": query,
        "role_play": role_play,
        "context": context,
        "inst_text": inst_text,
        "spc_type": 'llm_llama',      # 可以指定llm大模型类型(llm_llama, llm_glm)，不过每次切换都需要卸载/加载，尽量不要切换
        "max_num_sentence": max_num_sentence,
        "repetition_penalty": repetition_penalty
    }
    response = await post_retry(url, headers, data)
    if response.status_code == 200:
        resp_json = json.loads(response.text)
        return resp_json['text']
    else:
        return ""


async def tts_async(text, ref_name, out_name, spc_type, et_uuid=_ET_UUID_, language="english"):
    url = f"{HOST}/tts/tr"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    # "spc_type": "ov_v2"
    # "ref_name": ref_name      # 指定音色
    # 以下是chatTTS参数
    # "spc_type": "chat_tts"
    # "manual_seed": 0       # 指定音色: 414女，410男
    # "skip_refine_text": False # True表示自行插入语气
    data = {
        "et_uuid": et_uuid,
        "language": language,
        "text": text,
        "out_name": out_name,
        "spc_type": spc_type,
        "ref_name": ref_name,
        "manual_seed": 8,       # 1~22是预设音色
        "refine_prompt": "[oral_7][laugh_1][break_2]",
        "infer_prompt": "[speed_6]",
        "skip_refine_text": False
    }
    response = await post_retry(url, headers, data)
    if response.status_code == 200:
        resp_json = json.loads(response.text)
        return resp_json['path']
    else:
        return ""


async def sop_llm_tts_async(query, llm_type, role_play, context, inst_text, max_num_sentence,
                            out_name, ref_name, tts_type, et_uuid=_ET_UUID_, language="english"):
    url = f"{HOST}/llm/tts/tr"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "llm_param": {
            "et_uuid": et_uuid,
            "language": language,
            "use_history": True,
            "query": query,
            "role_play": role_play,
            "context": context,
            "inst_text": inst_text,
            "spc_type": llm_type,
            "max_num_sentence": max_num_sentence,
        },
        "tts_param": {
            "et_uuid": et_uuid,
            "language": language,
            "spc_type": tts_type,
            "out_name": out_name,
            "ref_name": ref_name,
            "manual_seed": ref_name,
            "refine_prompt": "[oral_1][laugh_1][break_1]",
            "skip_refine_text": False
        }
    }
    response = await post_retry(url, headers, data)
    if response.status_code == 200:
        resp_json = json.loads(response.text)
        return resp_json['text'], resp_json['path']
    else:
        return "", ""
