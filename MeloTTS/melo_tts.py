import os
import shutil

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from MeloTTS.melo.api import TTS
from et_base import ET_TTS

LANGUAGE_DICT = {
    "English": {"language": "EN_NEWEST", "path": "English-v3", "speaker": 0},
    "French": {"language": "FR", "path": "French", "speaker": 0},
    "Spanish": {"language": "ES", "path": "Spanish", "speaker": 0},
    "Chinese": {"language": "ZH", "path": "Chinese", "speaker": 0},
}


class Melo_tts(ET_TTS):
    def __init__(self, language='English'):
        super().__init__()
        lang_setting = LANGUAGE_DICT[language]
        # model setting
        from et_dirs import melo_tts_base, model_dir_base
        model_base = os.path.join(model_dir_base, os.path.basename(melo_tts_base))
        melo_models = os.path.join(model_base, f'models')
        config_path = f'{melo_models}{os.path.sep}{lang_setting["path"]}{os.path.sep}config.json'
        ckpt_path = f'{melo_models}{os.path.sep}{lang_setting["path"]}{os.path.sep}checkpoint.pth'
        self.model = TTS(language=lang_setting['language'], config_path=config_path, ckpt_path=ckpt_path, device=device)
        self.speaker_id = lang_setting['speaker']

    def tts(self, text: str, ref_speaker: str, **kwargs):
        # 推理
        if "speed" in kwargs:
            speed = kwargs["speed"]
        else:
            speed = 1.0
        # Run tts converter
        src_path = f'{self.output_dir}{os.path.sep}tmp.wav'
        self.model.tts_to_file(text, self.speaker_id, src_path, speed=speed, quiet=True)
        # return result
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(src_path, kwargs['output'])):
            shutil.copyfile(src_path, kwargs['output'])
            os.remove(src_path)
            src_path = kwargs['output']
        return src_path
