import os
import shutil
from et_base import check_multi_head_attention
from et_base import ET_TTS
from et_dirs import sovits_svc_base, model_dir_base
# 设置模型
base_dir = os.path.join(model_dir_base, os.path.basename(sovits_svc_base))
model_dir = os.path.join(base_dir, f'logs{os.path.sep}44k')
from infer_noui import vc_fn2, modelAnalysis
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SoVits_Svc(ET_TTS):
    def __init__(self, **kwargs):
        super().__init__()
        # 加载模型
        model_name = None
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        if not model_name or model_name.strip() == '':
            model_name = 'G_200000.pth'
        model_path = os.path.join(model_dir, model_name)
        assert os.path.exists(model_path)
        config_name = None
        if 'config_name' in kwargs:
            config_name = kwargs['config_name']
        if not config_name or config_name.strip() == '':
            config_name = 'config.json'
        config_path = os.path.join(model_dir, config_name)
        assert os.path.exists(config_path)
        speaker_id, msg = modelAnalysis(
            model_path=model_path, config_path=config_path, cluster_model_path=None, device='Auto',
            enhance=False, diff_model_path=None, diff_config_path=None, only_diffusion=False, use_spk_mix=False,
            local_model_enabled=False, local_model_selection=None
        )
        self.speaker_id = speaker_id
        # # model setting
        # parent_dir = os.path.join(base_dir, '..')
        # melo_dir = f'{parent_dir}{os.path.sep}OpenVoice_V2'
        # melo_models = f'{melo_dir}{os.path.sep}MeloTTS{os.path.sep}models'
        # config_path = f'{melo_models}{os.path.sep}English-v3{os.path.sep}config.json'
        # ckpt_path = f'{melo_models}{os.path.sep}English-v3{os.path.sep}checkpoint.pth'
        # self.model = TTS(language='EN_NEWEST', config_path=config_path, ckpt_path=ckpt_path, device=device)

    def tts(self, text: str, ref_speaker: str, **kwargs):
        check_multi_head_attention()
        # 参数处理
        if 'lang' in kwargs:
            lang = kwargs['lang']
        else:
            lang = 'Auto'
        if 'gender' in kwargs:
            gender = kwargs['gender']
        else:
            gender = '男'
        if 'f0_up_key' in kwargs:
            f0_up_key = kwargs['f0_up_key']
        else:
            f0_up_key = 0
        if 'f0_predictor' in kwargs:
            f0_predictor = kwargs['f0_predictor']
        else:
            f0_predictor = 'rmvpe'
        if 'output_format' in kwargs:
            output_format = kwargs['output_format']
        else:
            output_format = 'mp3'
        # 开始推理
        ret, audio_path = vc_fn2(text, lang, gender,
                                 0, 0, self.speaker_id, output_format, f0_up_key, True, 0, -40, 0.4,
                                 0.5, 0, 0, 0.75, f0_predictor, 0, 0.05,
                                 100, False, False, 0)
        # out = kwargs['output']
        # self.model.tts_to_file(text, 0, out, speed=1.0)
        # ret, audio_path = vc_fn3(out, self.speaker_id, output_format, f0_up_key, True, 0, -40, 0.4,
        #                          0.5, 0, 0, 0.75, f0_predictor, 0, 0.05,
        #                          100, False, False, 0)
        # 返回tts结果
        if ret == 'Success':
            if 'output' in kwargs and (not os.path.exists(kwargs['output']) or not os.path.samefile(audio_path, kwargs['output'])):
                shutil.copyfile(audio_path, kwargs['output'])
                audio_path = kwargs['output']
            return audio_path
        return None
