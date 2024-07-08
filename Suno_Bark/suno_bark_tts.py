import os.path
import shutil

from nltk.data import load
import numpy as np
import torch

from Suno_Bark.bark import semantic_to_waveform
from Suno_Bark.bark.generation import generate_text_semantic

device = "cuda:0" if torch.cuda.is_available() else "cpu"
from bark import SAMPLE_RATE, preload_models, generate_audio
from scipy.io import wavfile

from et_base import ET_TTS
from et_dirs import suno_bark_base, model_dir_base
model_dir = os.path.join(model_dir_base, os.path.basename(suno_bark_base), 'models')
os.environ['XDG_CACHE_HOME'] = os.path.join(model_dir, 'bark_v0')


class SunoAI_Bark(ET_TTS):
    def __init__(self, voice_preset, language="english"):
        super().__init__()
        preload_models()
        nltk_dir = os.path.join(model_dir, 'nltk_data')
        tokenizer_dir = os.path.join(nltk_dir, 'tokenizers', 'punkt', f'{language}.pickle')
        self.nltk_tokenizer = load(f'file:{tokenizer_dir}', format='pickle')
        self.voice_preset = voice_preset

    def tts(self, text: str, ref_speaker: str, **kwargs):
        voice_preset = self.voice_preset
        if 'voice_preset' in kwargs:
            voice_preset = kwargs['voice_preset']
        # save_path = self.tts_v1(text, voice_preset, **kwargs)
        save_path = self.tts_v2(text, voice_preset, **kwargs)
        return save_path

    def tts_v1(self, text: str, voice_preset: str, **kwargs):
        sentences = self.nltk_tokenizer.tokenize(text)
        pieces = []
        silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype='float32')
        for sentence in sentences:
            audio_array = generate_audio(sentence, history_prompt=voice_preset)
            pieces += [audio_array, silence.copy()]
        audio_array = np.concatenate(pieces, dtype='float32')
        save_path = f'{self.output_dir}{os.path.sep}output.wav'
        wavfile.write(save_path, SAMPLE_RATE, audio_array)
        # return result
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(save_path, kwargs['output'])):
            shutil.copyfile(save_path, kwargs['output'])
            os.remove(save_path)
            save_path = kwargs['output']
        return save_path

    def tts_v2(self, text: str, voice_preset: str, **kwargs):
        sentences = self.nltk_tokenizer.tokenize(text)
        pieces = []
        silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype='float32')
        for sentence in sentences:
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=voice_preset,
                temp=0.6,
                min_eos_p=0.05,  # this controls how likely the generation is to end
            )
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=voice_preset, )
            pieces += [audio_array, silence.copy()]
        audio_array = np.concatenate(pieces, dtype='float32')
        save_path = f'{self.output_dir}{os.path.sep}output.wav'
        wavfile.write(save_path, SAMPLE_RATE, audio_array)
        # return result
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(save_path, kwargs['output'])):
            shutil.copyfile(save_path, kwargs['output'])
            os.remove(save_path)
            save_path = kwargs['output']
        return save_path


if __name__ == '__main__':
    tts = SunoAI_Bark('v2/en_speaker_6', language='english')
    text = """
        Hey, have you heard about this new text-to-audio model called "Bark"? 
        Apparently, it's the most realistic and natural-sounding text-to-audio model 
        out there right now. People are saying it sounds just like a real person speaking. 
        I think it uses advanced machine learning algorithms to analyze and understand the 
        nuances of human speech, and then replicates those nuances in its own speech output. 
        It's pretty impressive, and I bet it could be used for things like audiobooks or podcasts. 
        In fact, I heard that some publishers are already starting to use Bark to create audiobooks. 
        It would be like having your own personal voiceover artist. I really think Bark is going to 
        be a game-changer in the world of text-to-audio technology.
    """.replace("\n", " ").strip()
    from et_base import timer
    with timer('sunoai-bark'):
        tts.tts(text, '')
