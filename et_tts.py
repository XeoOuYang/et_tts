import os
import shutil

from ChatTTS.chattts_tts import ChatTTS
from OpenVoice_V2.openvoicev2_tts import OpenVoiceV2_TTS
from et_base import yyyymmdd

TTS_INSTANCE = {
    'ov_v2': OpenVoiceV2_TTS(),
    'chat_tts': ChatTTS(),
}


def tts_chat_tts(text: str, ref_speaker, **kwargs):
    if 'chat_tts' not in TTS_INSTANCE:
        TTS_INSTANCE['chat_tts'] = ChatTTS()
    tts = TTS_INSTANCE['chat_tts']
    ret_path = tts.tts(text=text, ref_speaker=f'{ref_speaker}', manual_seed=ref_speaker, **kwargs)
    # resemble增强
    from et_base import timer
    with timer('resemble_enhance'):
        base_dir = os.path.dirname(ret_path)
        name, suffix = os.path.splitext(os.path.basename(ret_path))
        output = os.path.join(base_dir, f'{name}_resemble{suffix}')
        from Resemble_Enhance.enhance_resemble import Enhance_Resemble
        ret_path = Enhance_Resemble().denoise(ret_path, output=output)
    return ret_path


def tts_ov_v2(text: str, ref_speaker: str, **kwargs):
    if 'ov_v2' not in TTS_INSTANCE:
        TTS_INSTANCE['ov_v2'] = OpenVoiceV2_TTS()
    tts = TTS_INSTANCE['ov_v2']
    ret_path = tts.tts(text=text, ref_speaker=ref_speaker, **kwargs)
    return ret_path


def tts_gpt_sovits(text: str, ref_speaker: str, **kwargs):
    from GPT_SoVITS.gpt_sovits_tts import Gpt_SoVits
    if 'gpt_sovits' not in TTS_INSTANCE:
        TTS_INSTANCE['gpt_sovits'] = Gpt_SoVits(language='英文')
    tts = TTS_INSTANCE['gpt_sovits']
    ret_path = tts.tts(text=text, ref_speaker=ref_speaker, **kwargs)
    return ret_path


def rvc_convert(audio_path: str, **kwargs):
    from RVC_Converter.rvc_converter import Rvc_Converter
    if 'rvc_cvt' not in TTS_INSTANCE:
        TTS_INSTANCE['rvc_cvt'] = Rvc_Converter(model_name='et_man_role0.pth', speaker_id=0, f0_up_key=0)
    cvt = TTS_INSTANCE['rvc_cvt']
    ret_path = cvt.convert(audio_path=audio_path, **kwargs)
    return ret_path


def tts_sovits_svc(text: str, ref_speaker: str, **kwargs):
    from SoVITS_SVC.sovits_svc_tts import SoVits_Svc
    if 'sovits_svc' not in TTS_INSTANCE:
        TTS_INSTANCE['sovits_svc'] = SoVits_Svc()
    tts = TTS_INSTANCE['sovits_svc']
    ret_path = tts.tts(text=text, ref_speaker=ref_speaker, **kwargs)
    return ret_path


def tts_sunoai_bark(text: str, ref_speaker: str, **kwargs):
    from Suno_Bark.suno_bark_tts import SunoAI_Bark
    if 'sunoai_bark' not in TTS_INSTANCE:
        TTS_INSTANCE['sunoai_bark'] = SunoAI_Bark()
    tts = TTS_INSTANCE['sunoai_bark']
    ret_path = tts.tts(text=text, ref_speaker=ref_speaker, **kwargs)
    return ret_path


def read_txt(file: str):
    with open(file, 'r', encoding='utf8', errors='ignore') as txt_file:
        txt_str = txt_file.read()
    return txt_str.replace('**', '')


def read_json(file: str):
    import json
    with open(file, 'r', encoding='utf8', errors='ignore') as txt_file:
        json_dict = json.loads(txt_file.read())
    return json_dict


def tts(input_text, ref_audio, output_path, **kwargs):
    """
    # input_text：需要转换的文本内容
    # ref_audio：音频的目标音色
    # output_name：tts输出wav音频地址
    # se_style：感情配置，暂无使用
    """
    to_del = []
    name, _ = os.path.splitext(os.path.basename(output_path))
    from et_dirs import outputs_v2
    # output = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{name}_chat.wav')
    # tts_path = tts_chat_tts(text=input_text, ref_speaker=5656, output=output, **kwargs)
    # to_del.append(output)
    output = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{name}_ov_v2.wav')
    tts_path = tts_ov_v2(text=input_text, ref_speaker=ref_audio, speed=1.0, output=output, **kwargs)
    to_del.append(output)
    # output = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{name}_bark.wav')
    # tts_path = tts_sunoai_bark(text=input_text, ref_speaker=ref_audio, output=output, **kwargs)
    # to_del.append(output)
    # output = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{name}_gpt.wav')
    # tts_path = tts_gpt_sovits(text=input_text, ref_speaker=ref_audio, output=output)
    # to_del.append(output)
    # output = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{name}_svc.wav')
    # tts_path = tts_sovits_svc(text=input_text, ref_speaker=ref_audio, output=output)
    # to_del.append(output)
    # output = os.path.join(outputs_v2, f'{yyyymmdd}{os.path.sep}{name}_rvc.wav')
    # tts_path = rvc_convert(tts_path, output=output)
    # to_del.append(output)
    # 复制一份数据
    if not os.path.exists(output_path) or not os.path.samefile(output_path, tts_path):
        shutil.copyfile(tts_path, output_path)
    # 删除临时文件
    for tmp in to_del:
        if os.path.exists(tmp) and not os.path.samefile(tmp, output_path): os.remove(tmp)
    return output_path


if __name__ == '__main__':
    # todo: 从云读取
    # 从文件读取内容
    # text_str = read_txt(os.path.abspath(f'resources{os.path.sep}20240521.txt'))
    # text_list = [text_str.strip().replace('\n\n', '\n').strip('\n')]
    json_array = read_json(os.path.abspath(f'resources{os.path.sep}20240522.json'))
    text_list = [item['text'].strip().replace('\n\n', '\n').strip('\n') for item in json_array]
    ref_speaker = os.path.abspath(f'resources{os.path.sep}man_role0_ref.wav')
    output_dir = os.path.abspath(f'outputs_v2{os.path.sep}{yyyymmdd}{os.path.sep}20240522')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    text_list = [text_list[0]]
    # 开始转换
    for text_idx, text_str in enumerate(text_list):
        output_name = os.path.join(output_dir, f'tts_{text_idx}.wav')
        # if not os.path.exists(output_name):
        #     tts_result = tts(text_str, ref_speaker, output_name)
        tts_result = tts_chat_tts(text_str, 414, output=output_name)
        print(tts_result)
