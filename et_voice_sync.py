import asyncio
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from et_dirs import resources
LANGUAGE = {
    'en': 'English', 'sp': 'Spanish', 'zh': 'Chinese'
}
from faster_whisper import WhisperModel
from et_http_api import llm_async, tts_async

def asr(audio, lang) -> str:
    model = WhisperModel(model_size_or_path="large-v3", device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True,
                                      vad_parameters=dict(min_silence_duration_ms=700), word_timestamps=True)
    text_lang = "".join([segment.text for segment in segments])
    return text_lang

def translate(text, from_lang, to_lang) -> str:
    from_lang_name = LANGUAGE[from_lang]
    to_lang_name = LANGUAGE[to_lang]
    output_format = 'result should in JSON format {"output": translate result}, JSON only'
    llm_text = llm_async(query=text,
                         role_play=f'You are an expert of language, good at translating from {from_lang_name} to {to_lang_name}.',
                         context='',
                         inst_text=f'Please translate bellow text from {from_lang_name} to {to_lang_name}, {output_format}.',
                         max_num_sentence=16, language=None)
    llm_text = asyncio.run(llm_text)
    try:
        json_dict = json.loads(llm_text)
        return json_dict['output']
    except:
        return llm_text

def tts(text, speaker, lang) -> str:
    audio_path = tts_async(text, speaker, f'{lang}', spc_type='ov_v2', language=LANGUAGE[lang])
    audio_path = asyncio.run(audio_path)
    return audio_path

def play(audio):
    from sounddevice_wrapper import play_audio_async, SOUND_DEVICE_INDEX
    procedure = play_audio_async(audio, device=SOUND_DEVICE_INDEX[0])
    procedure.join()

if __name__ == '__main__':
    from et_base import timer
    # source = os.path.join(resources, 'example_reference.mp3')
    source = os.path.join(resources, '88795527.mp3')
    from_lang = 'zh'
    to_lang = 'sp'
    with timer('asr'):
        text = asr(source, lang=from_lang)
    print('asr.text ==>', text)
    with timer('translate'):
        text = translate(text, from_lang=from_lang, to_lang=to_lang)
    print('translate.text ==>', text)
    with timer('tts'):
        target = tts(text, 'example_reference', to_lang)
    print('tts.target ==>', target)
    play(target)
