import shutil

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openvoice.api import BaseSpeakerTTS
from et_base import check_multi_head_attention
from et_base import ET_TTS
from MeloTTS.melo_tts import Melo_tts
import os
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 设置模型
from et_dirs import ov_v2_base
base_dir = ov_v2_base

SES_DICT = {
    "English": "EN_NEWEST",
    "French": "FR",
    "Spanish": "ES",
    "Chinese": "ZH",
}


class OpenVoiceV2_TTS(ET_TTS):
    def __init__(self, lang='English'):
        super().__init__()
        from et_dirs import model_dir_base
        ckpt_dir = os.path.join(model_dir_base, f'OpenVoice_V2{os.path.sep}checkpoints_v2')
        ckpt_converter = os.path.join(ckpt_dir, f'converter')
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}{os.path.sep}config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}{os.path.sep}checkpoint.pth')
        self.tone_color_converter = tone_color_converter
        self.melo_tts = Melo_tts(language=lang)
        self.speaker_key = SES_DICT[lang].lower().replace('_', '-')
        pth_ses = os.path.join(ckpt_dir, f'base_speakers{os.path.sep}ses{os.path.sep}{self.speaker_key}.pth')
        self.source_se = torch.load(pth_ses, map_location=device)
        # v1
        # base_dir_v1 = os.path.join(model_dir_base, f'OpenVoice_V2{os.path.sep}checkpoints')
        # pth_ses_v1 = os.path.join(base_dir_v1, f'base_speakers{os.path.sep}EN')
        # # default, whispering, shouting, excited, cheerful, terrified, angry, sad, friendly
        # self.en_source_se = {
        #     'default': torch.load(f'{pth_ses_v1}{os.path.sep}en_default_se.pth').to(device),
        #     'style': torch.load(f'{pth_ses_v1}{os.path.sep}en_style_se.pth').to(device)
        # }
        # self.en_base_speaker_tts = BaseSpeakerTTS(f'{pth_ses_v1}{os.path.sep}config.json', device=device)
        # self.en_base_speaker_tts.load_ckpt(f'{pth_ses_v1}{os.path.sep}checkpoint.pth')

    def tts(self, text: str, ref_speaker: str, **kwargs):
        from et_base import text_normalize
        text = text_normalize(text, True)
        # return self.tts_v1(text, ref_speaker, **kwargs)
        return self.tts_v2(text, ref_speaker, **kwargs)

    # def tts_v1(self, text: str, ref_speaker: str, **kwargs):
    #     se_style = 'default'
    #     if 'se_style' in kwargs:
    #         se_style = kwargs['se_style']
    #     source_se = self.en_source_se['default'] if se_style == 'default' else self.en_source_se['style']
    #     # Run tts converter
    #     src_path = f'{self.output_dir}{os.path.sep}tmp.wav'
    #     self.en_base_speaker_tts.tts(text=text, output_path=src_path, speaker=se_style, language='English')
    #     # Run the tone color converter
    #     save_path = f'{self.output_dir}{os.path.sep}output.wav'
    #     target_se, audio_name = se_extractor.get_se(ref_speaker, self.tone_color_converter, vad=False)
    #     encode_message = "@MyShell"
    #     self.tone_color_converter.convert(
    #         audio_src_path=src_path,
    #         src_se=source_se,
    #         tgt_se=target_se,
    #         output_path=save_path,
    #         message=encode_message)
    #     # return result
    #     if 'output' in kwargs and (not os.path.exists(kwargs['output'])
    #                                or not os.path.samefile(save_path, kwargs['output'])):
    #         shutil.copyfile(save_path, kwargs['output'])
    #         os.remove(save_path)
    #         save_path = kwargs['output']
    #     return save_path

    def tts_v2(self, text: str, ref_speaker: str, **kwargs):
        check_multi_head_attention()
        # 推理
        if "speed" in kwargs:
            speed = kwargs["speed"]
        else:
            speed = 1.0
        # Run tts converter
        src_path = f'{self.output_dir}{os.path.sep}tmp.wav'
        self.melo_tts.tts(text, '', output=src_path, speed=speed)

        # Run the tone color converter
        save_path = f'{self.output_dir}{os.path.sep}output.wav'
        target_se, audio_name = se_extractor.get_se(ref_speaker, self.tone_color_converter, vad=False)
        encode_message = "@MyShell"
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message)

        # return result
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(save_path, kwargs['output'])):
            shutil.copyfile(save_path, kwargs['output'])
            os.remove(save_path)
            save_path = kwargs['output']
        return save_path
