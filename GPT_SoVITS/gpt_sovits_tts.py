import shutil

from et_base import ET_TTS
import os

from et_dirs import gpt_sovits_base, model_dir_base
# 设置模型
base_dir = os.path.join(model_dir_base, os.path.basename(gpt_sovits_base))
pretrained_dir = os.path.join(base_dir, 'pretrained_models')
bert_path = os.path.join(pretrained_dir, f'chinese-roberta-wwm-ext-large')
os.environ.setdefault('bert_path', bert_path)
cnhubert_base_path = os.path.join(pretrained_dir, f'chinese-hubert-base')
os.environ.setdefault('cnhubert_base_path', cnhubert_base_path)
gpt_path = os.path.join(pretrained_dir, f's1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')
os.environ.setdefault('gpt_path', gpt_path)
sovits_dir = os.path.join(base_dir, 'SoVITS_weights')
sovits_path = os.path.join(sovits_dir, f'et_man_role0_e8_s80.pth')
os.environ.setdefault('sovits_path', sovits_path)
# 不允许半精度
os.environ.setdefault("is_half", "True")

import soundfile as sf
import inference_cli
from inference_cli import get_tts_wav

from torch.nn import functional as F
from AR.modules.patched_mha_with_cache import multi_head_attention_forward_patched
_set_multi_head_attention_forward_ = multi_head_attention_forward_patched


def check_multi_head_attention():
    if F.multi_head_attention_forward != _set_multi_head_attention_forward_:
        F.multi_head_attention_forward = _set_multi_head_attention_forward_

# Generative Pre-trained Transformer: GPT
class Gpt_SoVits(ET_TTS):
    def __init__(self, language: str = '英文', sovits_model: str = None):
        super().__init__()
        self.language = language
        # 加载模型
        if sovits_model and inference_cli.sovits_path != sovits_model:
            inference_cli.change_sovits_weights(sovits_model)

    def tts(self, text: str, ref_speaker: str, **kwargs):
        check_multi_head_attention()
        # 推理
        output_list = get_tts_wav(
            ref_wav_path=ref_speaker,
            prompt_text=None,
            prompt_language=self.language,
            text=text,
            text_language=self.language,
            how_to_cut='按英文句号.切',
            ref_free=True
        )
        # 保存
        result_list = list(output_list)
        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = os.path.join(self.output_dir, "output.wav")
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            # return result
            if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                       or not os.path.samefile(output_wav_path, kwargs['output'])):
                shutil.copyfile(output_wav_path, kwargs['output'])
                os.remove(output_wav_path)
                output_wav_path = kwargs['output']
            return output_wav_path
        else:
            return None

