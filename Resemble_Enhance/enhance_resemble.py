import os
import shutil

import torch
import torchaudio
from scipy.io import wavfile

from resemble_enhance.enhancer.inference import enhance
from resemble_enhance.enhancer.inference import denoise

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from et_base import ET_BASE


class Enhance_Resemble(ET_BASE):
    def __init__(self, nfe=32, solver='midpoint', lambd=0.5, tau=0.5):
        super().__init__()
        """
        nfe: 1~128
        solver: 'midpoint', 'rk4', 'euler'
        lambd: 0~1
        tau: 0~1
        """
        self.solver = solver.lower()
        self.nfe = int(nfe)
        self.tau = tau
        self.lambd = lambd
        from et_dirs import model_dir_base, resemble_enhance_base
        base_name = os.path.basename(resemble_enhance_base)
        self.run_dir = os.path.join(model_dir_base, f'{base_name}{os.path.sep}model_repo')

    def resemble(self, path, **kwargs):
        return self.enhance(self.denoise(path, **kwargs), **kwargs)

    def denoise(self, path, **kwargs):
        if path is not None:
            dwav, sr = torchaudio.load(str(path))
            dwav = dwav.mean(dim=0)
            # 降噪
            wav, new_sr = denoise(dwav, sr, device, run_dir=self.run_dir)
            wav = wav.cpu().numpy()
            # 写入文件
            name, suffix = os.path.splitext(os.path.basename(path))
            output_path = f'{self.output_dir}{os.path.sep}{name}_denoise{suffix}'
            wavfile.write(output_path, new_sr, wav)
        else:
            output_path = path
        # 返回结果
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(output_path, kwargs['output'])):
            shutil.copyfile(output_path, kwargs['output'])
            os.remove(output_path)
            output_path = kwargs['output']
        return output_path

    def enhance(self, path, **kwargs):
        if path is not None:
            dwav, sr = torchaudio.load(str(path))
            dwav = dwav.mean(dim=0)
            # 增强
            wav, new_sr = enhance(dwav, sr, device, nfe=self.nfe, solver=self.solver, lambd=self.lambd, tau=self.tau,
                                  run_dir=self.run_dir)
            wav = wav.cpu().numpy()
            # 写入文件
            name, suffix = os.path.splitext(os.path.basename(path))
            output_path = f'{self.output_dir}{os.path.sep}{name}_enhance{suffix}'
            wavfile.write(output_path, new_sr, wav)
        else:
            output_path = path
        # 返回结果
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(output_path, kwargs['output'])):
            shutil.copyfile(output_path, kwargs['output'])
            os.remove(output_path)
            output_path = kwargs['output']
        return output_path
