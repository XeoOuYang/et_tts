import shutil

from et_base import check_multi_head_attention
from et_base import ET_BASE
from pathlib import Path
from scipy.io import wavfile
from rvc.modules.vc.modules import VC
import os

from et_dirs import rvc_cvt_base, model_dir_base
# 设置模型
base_dir = os.path.join(model_dir_base, os.path.basename(rvc_cvt_base))
assets_dir = os.path.join(base_dir, 'assets')
weights_dir = os.path.join(assets_dir, f'weights')
huber_dir = os.path.join(assets_dir, f'hubert')


from dotenv import load_dotenv


def init_rvc_env_path():
    env_file_path = os.path.join(base_dir, ".env")
    if not os.path.exists(env_file_path):
        default_values = {
            "weight_root": os.path.join(base_dir, f"assets{os.path.sep}weights"),
            "weight_uvr5_root": os.path.join(base_dir, f"assets{os.path.sep}uvr5_weights"),
            "save_uvr_path": "",
            "index_root": os.path.join(base_dir, "logs"),
            "rmvpe_root": os.path.join(base_dir, f"assets{os.path.sep}rmvpe"),
            "hubert_path": os.path.join(base_dir, f"assets{os.path.sep}hubert{os.path.sep}hubert_base.pt"),
            "pretrained": os.path.join(base_dir, f"assets{os.path.sep}pretrained_v2"),
            "TEMP": "",
        }
        with open(env_file_path, "w") as env_file:
            for key, value in default_values.items():
                env_file.write(f"{key}={value}\n")
    # 初始化环境变量
    load_dotenv(env_file_path)


# 导入文件直接调用
init_rvc_env_path()


class Rvc_Converter(ET_BASE):
    def __init__(self, **kwargs):
        super().__init__()
        self.vc = VC()
        model_name = None
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
        if not model_name or model_name.strip() == '':
            model_name = 'et_man_role0.pth'
        model_path = os.path.join(weights_dir, model_name)
        self.vc.get_vc(model_path)
        speaker_id = 0
        if 'speaker_id' in kwargs:
            speaker_id = kwargs['speaker_id']
        self.speaker_id = speaker_id
        f0_up_key = 0
        if 'f0_up_key' in kwargs:
            f0_up_key = kwargs['f0_up_key']
        self.f0_up_key = f0_up_key
        hubert_name = None
        if 'hubert_name' in kwargs:
            hubert_name = kwargs['hubert_name']
        if not hubert_name or hubert_name.strip() == '':
            hubert_name = 'hubert_base.pt'
        self.hubert_path = os.path.join(huber_dir, hubert_name)
        os.environ.setdefault("hubert_path", self.hubert_path)

    def convert(self, audio_path: str, **kwargs):
        check_multi_head_attention()
        # 推理
        tgt_sr, audio_opt, times, _ = self.vc.vc_inference(
            self.speaker_id, Path(audio_path), f0_up_key=self.f0_up_key, hubert_path=self.hubert_path
        )
        name, suffix = os.path.splitext(os.path.basename(audio_path))
        output_path = f'{self.output_dir}{os.path.sep}{name}_rvc{suffix}'
        wavfile.write(output_path, tgt_sr, audio_opt)
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(output_path, kwargs['output'])):
            shutil.copyfile(output_path, kwargs['output'])
            os.remove(output_path)
            output_path = kwargs['output']
        return output_path
