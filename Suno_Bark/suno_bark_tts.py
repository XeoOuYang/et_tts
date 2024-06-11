import os.path
import shutil
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from transformers import AutoProcessor, BarkModel
from scipy.io import wavfile
SAMPLE_RATE = 24_000

from et_base import ET_TTS


class SunoAI_Bark(ET_TTS):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark").to(device)
        self.voice_preset = "v2/en_speaker_6"
        # self.voice_preset = "v2/en_speaker_9"

    def tts(self, text: str, ref_speaker: str, **kwargs):
        voice_preset = self.voice_preset
        if 'voice_preset' in kwargs:
            voice_preset = kwargs['voice_preset']
        inputs = self.processor(text, voice_preset).to(device)
        outputs = self.model.generate(**inputs)
        outputs = outputs.cpu().numpy().squeeze()
        save_path = f'{self.output_dir}{os.path.sep}output.wav'
        wavfile.write(save_path, SAMPLE_RATE, outputs)
        # return result
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(save_path, kwargs['output'])):
            shutil.copyfile(save_path, kwargs['output'])
            os.remove(save_path)
            save_path = kwargs['output']
        return save_path


if __name__ == '__main__':
    tts = SunoAI_Bark()
    text = 'Micro Ingredients 7 in 1 full spectrum hydrolyzed collagen peptides powder.'
    output = f'output.wav'
    from et_base import timer
    with timer('sunoai-bark'):
        tts.tts(text, '', output=output)
