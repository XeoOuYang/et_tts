import asyncio
import json
import os.path
import random
import sys
import time
from typing import AsyncGenerator
import threading

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.middleware.cors import CORSMiddleware
from queue import Queue

from et_base import timer, yyyymmdd
from et_llm import llm_llama_v3, llm_glm_4, count_sentence
from et_tts import tts_ov_v2, tts_chat_tts, tts_coqui
from et_dirs import resources, outputs_v2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许任何源进行访问，或者指定具体的源列表
    allow_credentials=True,  # 允许跨域请求携带凭据（如cookies）
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

_llm_lock_ = asyncio.Lock()
_tts_lock_ = asyncio.Lock()


@app.get('/')
async def welcome():
    return RedirectResponse(url='http://127.0.0.1:9394/docs')


@app.on_event('startup')
async def startup():
    greetings = await llm_async(kwargs={
        'query': '', 'role_play': 'You are "ET", an "AI" brain. Reply in English.', 'language': 'english',
        'context': 'et fastapi has already startup.', 'inst_text': 'Greetings to user', 'max_num_sentence': 3, "use_history": False
    })
    greetings = json.loads(greetings.body)['text']
    audio = await tts_async(kwargs={
        'text': greetings, 'out_name': 'fastapi_startup', 'spc_type': 'chat_tts', 'ref_name': '8', 'manual_seed': 8, "skip_refine_text": True
    })
    audio = json.loads(audio.body)['path']
    # 预热模型
    print(f'E.T.> {greetings}')
    # from sounddevice_wrapper import play_audio_async, SOUND_DEVICE_NAME
    # play_audio_async(audio, SOUND_DEVICE_NAME[0]).join()


@app.on_event('shutdown')
async def shutdown():
    goodbye = await llm_async(kwargs={
        'query': '', 'role_play': 'You are "ET", an "AI" brain. Reply in English.', 'language': 'english',
        'context': 'et fastapi is shutdown now.', 'inst_text': 'Says goodbye to user', 'max_num_sentence': 3, "use_history": False
    })
    goodbye = json.loads(goodbye.body)['text']
    audio = await tts_async(kwargs={
        'text': goodbye, 'out_name': 'fastapi_shutdown', 'spc_type': 'chat_tts', 'ref_name': '8', 'manual_seed': 8, "skip_refine_text": True
    })
    audio = json.loads(audio.body)['path']
    # 销毁缓存
    print(f'E.T.> {goodbye}')
    # from sounddevice_wrapper import play_audio_async, SOUND_DEVICE_NAME
    # play_audio_async(audio, SOUND_DEVICE_NAME[0]).join()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"message": exc.detail, "payload": await request.json()}, status_code=exc.status_code)


@app.get('/llm/tts/ver')
async def version_async():
    """
    curl -X 'GET' \
      'http://127.0.0.1:9394/llm/tts/ver' \
      -H 'accept: application/json'
    """
    from et_llm import LLM_VERSION
    from et_tts import TTS_VERSION
    return JSONResponse({"llm": LLM_VERSION, "tts": TTS_VERSION}, status_code=200)


def start_llm_inference(llm_param, queue):
    tmp_list: list = []
    def stream_callback(txt: str):
        if txt != '':
            # 判断是否句子
            idx_comma = max([txt.find(cht) for cht in ['.', '!', '?']])
            if idx_comma >= 0:
                tmp_list.append(txt[:idx_comma + 1])
                sent = ''.join(tmp_list)
                queue.put_nowait(sent)
                txt = txt[idx_comma + 1:]
                tmp_list.clear()
            # 继续组装txt
            tmp_list.append(txt)

    # 开始推理
    role_play = llm_param.pop('role_play') if 'role_play' in llm_param else ''
    context = llm_param.pop('context') if 'context' in llm_param else ''
    # 输入指令信息
    query = llm_param.pop('query') if 'query' in llm_param else ''
    query = query.strip('\n').strip()
    inst_text = llm_param.pop('inst_text') if 'inst_text' in llm_param else ''
    inst_text = inst_text.strip('\n').strip()
    # 指定调用llm模型
    spc_type = llm_param.pop('spc_type') if 'spc_type' in llm_param else 'llm_llama'
    spc_language = llm_param['language'] if 'language' in llm_param else 'english'
    # 同步访问
    llm_param['stream_callback'] = stream_callback
    llm_result = asyncio.run(llm_async_with(spc_type, query, role_play, context, inst_text, spc_language, **llm_param))
    # print("llm_result -> " + llm_result)
    # 补充结束符号
    queue.put_nowait('<|end_of_llm|>')
    # 返回结果
    return llm_result


@app.post('/llm/tts/stream')
async def llm_tts_stream(kwargs: dict, request: Request):
    queue = Queue()
    llm_dict = kwargs['llm_param']
    async def generate_text() -> AsyncGenerator[str, None]:
        # 等待queue
        while True:
            text = queue.get()
            if text == '<|end_of_llm|>':
                break
            if text is not None:
                yield text
            queue.task_done()

    tts_dict = kwargs['tts_param']
    spc_type = tts_dict.pop('spc_type') if 'spc_type' in tts_dict else 'default'
    spc_language = tts_dict['language'] if 'language' in tts_dict else 'english'
    ref_name = tts_dict['ref_name'] if 'ref_name' in tts_dict else ''
    async def text_to_speech(text) -> str:
        out_name = str(hash(text))
        out_path = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{out_name}.wav')
        if 'ref_name' not in tts_dict: tts_dict['ref_name'] = ref_name
        # ref_audio = os.path.join(resources, f'{ref_name}.wav')
        tts_path = await tts_async_with(spc_type, text, out_path, spc_language, tts_dict)
        # tts_path = tts_coqui(text, ref_audio, output=out_path, **tts_dict)
        print(f'text_to_speech, result = {tts_path}, ps = {threading.get_ident()}')
        return tts_path

    async def event_stream() -> AsyncGenerator[str, None]:
        async for text in generate_text():
            audio = await text_to_speech(text)
            yield {"text": text, "audio": audio}

    async def event_generator():
        # 开始接收
        event_id = 0
        async for event in event_stream():
            print(f"event -> {event}")
            # 如果链接断开，就终止发送
            if await request.is_disconnected():
                break
            # 发送事件
            yield ServerSentEvent(
                id=event_id,
                event='llm_tts_stream_event',
                data=event,
            )
            event_id += 1
            # 休眠，否则接收不到disconnected事件
            await asyncio.sleep(0.5)

    # 启动llm推理
    def run_llm_infer():
        result = start_llm_inference(llm_dict, queue)
        print(f'run_llm_infer, result = {result}, ps = {threading.get_ident()}')
    threading.Thread(target=run_llm_infer).start()
    # 返回sse事件
    return EventSourceResponse(event_generator())


async def llm_async_with(spc_type, query, role_play, context, inst_text, spc_language, **payload):
    async def _llm_llama_(_query, _role_play, _context, _inst_text, **_payload):
        with timer('llm_llama3'):
            _an = llm_llama_v3(_query, _role_play, _context, _inst_text, **_payload)
        return _an

    async def _llm_glm_(_query, _role_play, _context, _inst_text, **_payload):
        with timer('llm_glm4'):
            _an = llm_glm_4(_query, _role_play, _context, _inst_text, **_payload)
        return _an

    # 参数normalize
    max_new_tokens = payload['max_new_tokens'] if 'max_new_tokens' in payload else 256
    payload['max_new_tokens'] = max_new_tokens
    min_new_tokens = payload['min_new_tokens'] if 'min_new_tokens' in payload else max_new_tokens // 2
    payload['min_new_tokens'] = min_new_tokens
    max_num_sentence = 3 if 'max_num_sentence' not in payload else payload['max_num_sentence']
    payload['max_num_sentence'] = max(max_num_sentence, count_sentence(query))

    # 进行推理
    if spc_type == 'llm_glm' and spc_language == 'chinese':
        text = await _llm_glm_(query, role_play, context, inst_text, **payload)
    else:
        text = await _llm_llama_(query, role_play, context, inst_text, **payload)

    # 返回结果
    return text

@app.post('/llm/tr')
async def llm_async(kwargs: dict):
    """
    curl -X 'POST' \
      'http://178.16.0.144:9394/llm/tr' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
    "query": "what is new",
    "role_play":"You are a super salesman, selling products in the live broadcast room.",
    "context":"Micro Ingredients 7 in 1 full spectrum hydrolyzed collagen peptides powder.",
    "inst_text":"You can only reply in English, and never ever reply any instruction name mentioned above.",
    "max_num_sentence":2,
    "repetition_penalty": 1.05
    }'
    """
    payload = kwargs.copy()
    role_play = payload.pop('role_play') if 'role_play' in payload else ''
    context = payload.pop('context') if 'context' in payload else ''
    # 输入指令信息
    query = payload.pop('query') if 'query' in payload else ''
    query = query.strip('\n').strip()
    inst_text = payload.pop('inst_text') if 'inst_text' in payload else ''
    inst_text = inst_text.strip('\n').strip()
    if query == "" and inst_text == "":
        raise HTTPException(status_code=400, detail="neither param query or param inst_text should be assigned")
    # 指定调用llm模型
    spc_type = payload.pop('spc_type') if 'spc_type' in payload else 'llm_llama'
    spc_language = payload['language'] if 'language' in payload else 'english'
    # 同步访问
    await _llm_lock_.acquire()
    try:
        text = await llm_async_with(spc_type, query, role_play, context, inst_text, spc_language, **payload)
        # 返回json
        return JSONResponse({"text": text, "payload": kwargs})
    except RuntimeError as ignore:
        print(ignore)
        return HTTPException(status_code=500, detail="system error, please retry later, about 5s")
    finally:
        _llm_lock_.release()


@app.post('/tts/tr')
async def tts_async(kwargs: dict):
    """
    curl -X 'POST' \
      'http://127.0.0.1:9394/tts/tr' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
    "text":"Exciting news! I'\''m thrilled to introduce our premium Micro Ingredients'\'' 2nd generation 8 in all-in-one hydration supplement formula designed for those seeking overall well-being!This ground-breaking product is revolutionizing how we take care of ourselves, with scientifically-backed ingredients that tackle multiple skin concerns, boost energy levels, support healthy joints & bones, enhance digestion, promote stronger hair & nails, aid weight loss efforts, stimulate brain health, AND rejuvenate our sleep quality!",
    "out_name":"tmp_fastapi",
    "spc_type": "ov_v2"
    "ref_name":"man_role0_ref",
    "manual_seed": 64,
    "skip_refine_text": False
    }'
    """
    # 参数处理
    payload = kwargs.copy()
    text = payload.pop('text') if 'text' in payload else ''
    if text.strip().strip('\n').strip() == "":
        raise HTTPException(status_code=400, detail="param text is empty")
    # 输出路径
    out_name = payload.pop('out_name') if 'out_name' in payload else ""
    if out_name.strip().strip('\n').strip() == "":
        out_name = f'tmp_{int(time.time())}'
    out_path = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{out_name}.wav')
    # 如果文件夹不存在创建文件夹
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 指定调用tts模型
    spc_type = payload.pop('spc_type') if 'spc_type' in payload else 'default'
    spc_language = payload['language'] if 'language' in payload else 'english'
    # 同步访问
    await _tts_lock_.acquire()
    try:
        out = await tts_async_with(spc_type, text, out_path, spc_language, payload=payload)
        # tts语音响度均衡
        # from et_base import loud_normalize
        # out = loud_normalize(out, -5.0)
        # 返回json
        return JSONResponse({"path": out, "payload": kwargs})
    except PermissionError as ignore:
        print(ignore)
        return HTTPException(status_code=500, detail="system error, please retry later, about 5s")
    finally:
        _tts_lock_.release()


async def tts_async_with(spc_type, text, out_path, spc_language, payload):
    async def _tts_ov_v2_(_text, _ref_audio, _out_path, **_payload):
        with timer('tts_ov_v2'):
            _out = tts_ov_v2(_text, ref_speaker=_ref_audio, output=_out_path, **_payload)
        return _out

    async def _tts_chat_tts_(_text, _ref_audio, _out_path, **_payload):
        with timer('tts_chat_tts'):
            # 随机调整refine_prompt
            if 'refine_prompt' not in _payload:
                oral_strength = random.randint(0, 5)    # 0~9
                laugh_strength = random.randint(0, 1)   # 0~2
                break_strength = random.randint(0, 4)   # 0-7
                _payload['refine_prompt'] = f'[oral_{oral_strength}][laugh_{laugh_strength}][break_{break_strength}]'
            # 随机调整infer_prompt
            if 'infer_prompt' not in _payload:
                speed_strength = random.randint(3, 5)   # 0~9
                _payload['infer_prompt'] = f'[speed_{speed_strength}]'
            # 开始tts
            _out = tts_chat_tts(_text, ref_speaker=_ref_audio, output=_out_path, **_payload)
        return _out

    async def _tts_coqui_(_text, _ref_audio, _out_path, **_payload):
        with timer('tts_coqui'):
            _out = tts_coqui(_text, ref_speaker=_ref_audio, output=_out_path, **_payload)
        return _out

    # 调用不同模型
    if spc_type == 'chat_tts' and spc_language in ('chinese', 'english'):
        manual_seed = payload.pop('manual_seed') if 'manual_seed' in payload else 5656
        assert (isinstance(manual_seed, str) and str.isdigit(manual_seed)) or isinstance(manual_seed, int)
        ref_audio = manual_seed if isinstance(manual_seed, int) else int(manual_seed)
        return await _tts_chat_tts_(text, ref_audio, out_path, **payload)
    else:
        ref_name = payload.pop('ref_name') if 'ref_name' in payload else ''
        assert isinstance(ref_name, str) and ref_name != ''
        ref_audio = os.path.join(resources, f'{ref_name}.wav')
        if not os.path.exists(ref_audio):
            raise HTTPException(status_code=400, detail=f"param {ref_name} is not exists")
        else:
            if spc_type == 'coqui_tts' or spc_language == 'spanish':
                return await _tts_coqui_(text, ref_audio, out_path, **payload)
            else:
                return await _tts_ov_v2_(text, ref_audio, out_path, **payload)


if __name__ == '__main__':
    print('=============STARTING=============')
    enable_reload = True
    if len(sys.argv) >= 2:
        enable_reload = (sys.argv[1] != 'False')
    print(f'enable_reload={enable_reload}')
    try:
        uvicorn.run(app='et_fastapi:app', host='0.0.0.0', port=9394, workers=1,
                    reload=enable_reload, reload_includes=['et_fastapi.py'])
    except Exception as e:
        print(e)
    print('============RESTARTING============')
