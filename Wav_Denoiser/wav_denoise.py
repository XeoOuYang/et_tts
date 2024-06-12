import os
import shutil

import torch
import torchaudio

from Wav_Denoiser.audio_denoiser.AudioDenoiser import AudioDenoiser

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from et_base import ET_BASE


class Wav_Denoise(ET_BASE):
    def __init__(self, model_name='Wav_Denoise/audio-denoiser-512-32-v1', num_iterations=100):
        super().__init__()
        from et_dirs import model_dir_base
        model_path = os.path.join(model_dir_base, model_name)
        self.denoiser = AudioDenoiser(model_name=model_path, device=device, num_iterations=num_iterations)

    def denoise(self, path, **kwargs):
        data, sr = torchaudio.load(path)
        denoised = self.denoiser.process_waveform(data, sr, auto_scale=False)
        sr = self.denoiser.model_sample_rate
        denoised = denoised.cpu()
        # 写入文件
        name, suffix = os.path.splitext(os.path.basename(path))
        output_path = f'{self.output_dir}{os.path.sep}{name}_denoise{suffix}'
        torchaudio.save(output_path, denoised, sample_rate=sr)
        # 返回结果
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(output_path, kwargs['output'])):
            shutil.copyfile(output_path, kwargs['output'])
            os.remove(output_path)
            output_path = kwargs['output']
        return output_path
