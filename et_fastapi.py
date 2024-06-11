import asyncio
import json
import os.path
import random
import sys
import time

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from et_base import timer, yyyymmdd
from et_llm import llm_llama_v3, llm_glm_4
from et_tts import tts_ov_v2, tts_chat_tts
from et_dirs import resources, outputs_v2
from sounddevice_wrapper import play_audio_async, SOUND_DEVICE_NAME

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
        'query': 'Greetings to user', 'role_play': 'You are ET, an AI brain',
        'context': 'fastapi startup', 'inst_text': 'Reply in English', 'max_num_sentence': 3
    })
    greetings = json.loads(greetings.body)['text']
    audio = await tts_async(kwargs={'text': greetings, 'ref_name': 'man_role0_ref', 'out_name': 'fastapi_startup'})
    audio = json.loads(audio.body)['path']
    # 预热模型
    print(f'E.T.> {greetings}')
    play_audio_async(audio, SOUND_DEVICE_NAME[0]).join()


@app.on_event('shutdown')
async def shutdown():
    goodbye = await llm_async(kwargs={
        'query': 'Says goodbye to user', 'role_play': 'You are ET, an AI brain',
        'context': 'fastapi startup', 'inst_text': 'Reply in English', 'max_num_sentence': 3
    })
    goodbye = json.loads(goodbye.body)['text']
    audio = await tts_async(kwargs={'text': goodbye, 'ref_name': 'man_role0_ref', 'out_name': 'fastapi_shutdown'})
    audio = json.loads(audio.body)['path']
    # 销毁缓存
    print(f'E.T.> {goodbye}')
    play_audio_async(audio, SOUND_DEVICE_NAME[0]).join()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"message": exc.detail, "payload": request.body()}, status_code=exc.status_code)


@app.post('/llm/tts/tr')
async def concat(kwargs: dict = None):
    """
    curl -X 'POST' \
      'http://127.0.0.1:9394/concat/llm/tts' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
    "llm_param": {
    "query": "what is new",
    "role_play":"You are a super salesman, selling products in the live broadcast room.",
    "context":"Micro Ingredients 7 in 1 full spectrum hydrolyzed collagen peptides powder.",
    "inst_text":"You can only reply in English, and never ever reply any instruction name mentioned above.",
    "max_num_sentence":2,
    "repetition_penalty": 1.05
    },
    "tts_param": {
    "text":"{llm_out_text}",
    "out_name":"tmp_fastapi",
    "spc_type": "ov_v2"
    "ref_name":"man_role0_ref",
    "manual_seed": 64,
    "skip_refine_text": False
    }
    }'
    """
    payload = kwargs.copy()
    assert 'llm_param' in payload and 'tts_param' in payload
    # 启动llm任务
    llm_param = payload.pop('llm_param')
    query = llm_param['query']
    llm_resp = await llm_async(**llm_param)
    if llm_resp.status_code != 200:
        return llm_resp
    # 继续tts任务
    tts_text = json.loads(llm_resp.body)['text']
    tts_param = payload.pop('tts_param')
    tts_param['query'] = query
    tts_param['text'] = tts_text
    return await tts_async(**tts_param)


@app.post('/llm/tr')
async def llm_async(kwargs: dict = None):
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
    async def _llm_llama_(_query, _role_play, _context, _inst_text, **_payload):
        with timer('llm_llama3'):
            _an = llm_llama_v3(_query, _role_play, _context, _inst_text, **_payload)
        return _an

    async def _llm_glm_(_query, _role_play, _context, _inst_text, **_payload):
        with timer('llm_glm4'):
            _an = llm_glm_4(_query, _role_play, _context, _inst_text, **_payload)
        return _an

    payload = kwargs.copy()
    query = payload.pop('query') if 'query' in payload else ''
    if query.strip().strip('\n').strip() == "":
        raise HTTPException(status_code=400, detail="param query is empty")
    role_play = payload.pop('role_play') if 'role_play' in payload else ''
    context = payload.pop('context') if 'context' in payload else ''
    # 对context进行数字转换
    from et_base import text_normalize
    context = text_normalize(context)
    # 输入指令信息
    inst_text = payload.pop('inst_text') if 'inst_text' in payload else ''
    # 指定调用llm模型
    spc_type = payload.pop('spc_type') if 'spc_type' in payload else 'llm_llama'
    # 同步访问
    await _llm_lock_.acquire()
    try:
        if spc_type == 'llm_glm':
            text = await _llm_glm_(query, role_play, context, inst_text, **payload)
        else:
            text = await _llm_llama_(query, role_play, context, inst_text, **payload)
        # 返回json
        return JSONResponse({"text": text, "payload": kwargs})
    except RuntimeError as ignore:
        print(ignore)
        return HTTPException(status_code=500, detail="system error, please retry later, about 5s")
    finally:
        _llm_lock_.release()


@app.post('/tts/tr')
async def tts_async(kwargs: dict = None):
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
    # 同步访问
    await _tts_lock_.acquire()
    try:
        out = await tts_async_with(spc_type, text, out_path, payload=payload)
        # 返回json
        return JSONResponse({"path": out, "payload": kwargs})
    except PermissionError as ignore:
        print(ignore)
        return HTTPException(status_code=500, detail="system error, please retry later, about 5s")
    finally:
        _tts_lock_.release()


async def tts_async_with(spc_type, text, out_path, payload: dict = None):
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
            # 打印参数
            print(_payload['refine_prompt'], _payload['infer_prompt'])
            _out = tts_chat_tts(_text, ref_speaker=_ref_audio, output=_out_path, **_payload)
        return _out
    # 调用不同模型
    if spc_type == 'chat_tts':
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
            return await _tts_ov_v2_(text, ref_audio, out_path, **payload)


if __name__ == '__main__':
    print('=============STARTING=============')
    enable_reload = True
    if len(sys.argv) >= 2:
        enable_reload = (sys.argv[1] != 'False')
    print(f'enable_reload={enable_reload}')
    try:
        uvicorn.run(app='et_fastapi:app', host='0.0.0.0', port=9394,
                    reload=enable_reload, reload_includes=['et_fastapi.py'])
    except Exception as e:
        print(e)
    print('============RESTARTING============')
