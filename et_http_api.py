import asyncio

import requests
import uuid
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json

HOST = 'http://127.0.0.1:9394'
_ET_UUID_ = uuid.uuid1().hex


async def post_retry(url, headers, data, stream=None):
    retry_times = 3  # 设置重试次数
    retry_backoff_factor = 2.0  # 设置重试间隔时间
    status_force_list = [400, 401, 403, 404, 500, 502, 503, 504]
    session = requests.Session()
    retry = Retry(total=retry_times, backoff_factor=retry_backoff_factor, status_forcelist=status_force_list)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session.post(url, headers=headers, json=data, stream=stream), session


async def ver_async():
    url = f"{HOST}/llm/tts/ver"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    return response.status_code == 200


async def llm_glm(query, role_play, context, inst_text, max_num_sentence, repetition_penalty=1.05,
                  spc_type='llm_glm', et_uuid=_ET_UUID_, language="english"):
    return await llm_async(query, role_play, context, inst_text, max_num_sentence,
                           repetition_penalty, spc_type, et_uuid, language)


async def llm_llama(query, role_play, context, inst_text, max_num_sentence, repetition_penalty=1.05,
                    spc_type='llm_llama',et_uuid=_ET_UUID_, language="english"):
    return await llm_async(query, role_play, context, inst_text, max_num_sentence,
                           repetition_penalty, spc_type, et_uuid, language)


async def llm_async(query, role_play, context, inst_text, max_num_sentence, repetition_penalty=1.05,
                    spc_type='llm_llama',et_uuid=_ET_UUID_, language="english"):
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
        "spc_type": spc_type,      # 可以指定llm大模型类型(llm_llama, llm_glm)，不过每次切换都需要卸载/加载，尽量不要切换
        "max_num_sentence": max_num_sentence,
        "repetition_penalty": repetition_penalty
    }
    response, _ = await post_retry(url, headers, data)
    if response.status_code == 200:
        resp_json = json.loads(response.text)
        return resp_json['text']
    else:
        return ""


async def tts_coqui(text, ref_name, out_name, spc_type='coqui_tts', et_uuid=_ET_UUID_, language="Spanish"):
    return await tts_async(text, ref_name, out_name, spc_type, et_uuid, language)


async def tts_chat(text, ref_name: int, out_name, spc_type='chat_tts', et_uuid=_ET_UUID_, language="Chinese"):
    return await tts_async(text, str(ref_name), out_name, spc_type, et_uuid, language)


async def tts_async(text, ref_name, out_name, spc_type='ov_v2', et_uuid=_ET_UUID_, language="English"):
    url = f"{HOST}/tts/tr"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    # "spc_type": "ov_v2", "chat_tts", "coqui_tts"
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
    }
    if spc_type == 'chat_tts':
        data['manual_seed'] = int(ref_name) # 1~22是预设音色
        data['refine_prompt'] = "[oral_7][laugh_1][break_2]"
        data['infer_prompt'] = "[speed_6]"
        data['skip_refine_text'] = False
    # http请求
    response, _ = await post_retry(url, headers, data)
    if response.status_code == 200:
        resp_json = json.loads(response.text)
        return resp_json['path']
    else:
        return ""


async def llm_tts_stream(query, llm_type, role_play, context, inst_text, max_num_sentence, repetition_penalty,
                         ref_name, out_name, tts_type, et_uuid=_ET_UUID_, language="english"):
    url = f"{HOST}/llm/tts/stream"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Connection": "keep-alive"
    }
    llm_param = {
        "et_uuid": et_uuid,
        "language": language,
        "use_history": True,
        "query": query,
        "role_play": role_play,
        "context": context,
        "inst_text": inst_text,
        "spc_type": llm_type,
        "max_num_sentence": max_num_sentence,
        "repetition_penalty": repetition_penalty
    }
    tts_param = {
        "et_uuid": et_uuid,
        "language": language,
        "spc_type": tts_type,
        "out_name": out_name,
        "ref_name": ref_name,
    }
    if tts_type == 'chat_tts':
        tts_param['manual_seed'] = int(ref_name)  # 1~22是预设音色
        tts_param['refine_prompt'] = "[oral_7][laugh_1][break_2]"
        tts_param['infer_prompt'] = "[speed_6]"
        tts_param['skip_refine_text'] = False
    # 请求
    request_data = {
        "llm_param": llm_param,
        "tts_param": tts_param
    }
    import time
    start = time.perf_counter()
    response, session = await post_retry(url, headers, request_data, stream=True)
    if response.status_code != 200: return ""
    for line in response.iter_lines(chunk_size=1024, decode_unicode=True, delimiter='\n'):
        if line.startswith('data:'):
            print(time.perf_counter()-start)
            print(line)
    # 关闭sse
    session.close()
    print(time.perf_counter()-start)


if __name__ == '__main__':
    # 测试一、llm调整&翻译
    query = """
Hey guys, welcome back! Today's all about feeling amazing from the inside out,
and I've got the perfect due for you: Micro Ingredient's Multi Collagen Peptides and Mushroom Coffee.
Trust me, these two are about to become your new best friends for a healthier, happier you!
    """.replace("\n", " ").strip()
    from_lang_name = 'English'
    to_lang_name = 'Spanish'
    role_play = (f'You are good at translating from {from_lang_name} to {to_lang_name}, you only outputs the final result.\n'
                 f'Example Input:\n'
                 f'Please modify bellow text before translating from {from_lang_name} to {to_lang_name}.\n'
                 f'Hello, nice to meet you.\n'
                 f'Example Output:\n'
                 f'Hola mucho gusto.')
    inst_text = f'Please modify bellow text before translating from {from_lang_name} to {to_lang_name}.'
    llm_result = asyncio.run(llm_llama(
        query=query, role_play=role_play, context='', inst_text=inst_text, max_num_sentence=16,
        language=to_lang_name.lower()
    ))
    print(llm_result)
    # try:
    #     tr_text = json.loads(llm_result)['output']
    # except JSONDecodeError:
    #     tr_text = None
    # assert tr_text is not None
    tr_text = llm_result
    # 测试二、coqui-tts转换
    tts_path = asyncio.run(tts_coqui(
        tr_text, ref_name='ref_spanish_59s', out_name='welcome_man_0_es', language="Spanish"
    ))
    print(tts_path)
    assert tts_path is not None and tts_path != ''
    # 测试三、播放tts
    # from sounddevice_wrapper import play_audio_async, SOUND_DEVICE_INDEX
    # play_audio_async(tts_path, SOUND_DEVICE_INDEX[1])
