import os
import sys
import time

import librosa
from torch.nn import functional as F

from ChatTTS.tools import text_normalize

_keep_multi_head_attention_forward_ = F.multi_head_attention_forward
yyyymmdd = time.strftime('%Y%m%d', time.localtime())


def check_multi_head_attention():
    if F.multi_head_attention_forward != _keep_multi_head_attention_forward_:
        F.multi_head_attention_forward = _keep_multi_head_attention_forward_


class ET_BASE:
    def __init__(self):
        from et_dirs import prj_dir
        self.output_dir = os.path.join(prj_dir, f'outputs_v2{os.path.sep}{yyyymmdd}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_dir = prj_dir


class ET_TTS(ET_BASE):
    def __init__(self):
        super().__init__()

    def tts(self, text: str, ref_speaker: str, **kwargs):
        pass


class ET_LLM(ET_BASE):
    def __init__(self):
        super().__init__()

    def is_load(self):
        pass

    def unload_model(self):
        pass

    def llm(self, query: str, context: str):
        pass


import re

def reserve_char_all(text):
    # 中文 + 英文
    # \u4e00-\u9fff\u3000-\u303f\uac00-\ud7af
    # \u0020-\u007e\u00a0-\u00ff
    # 标点符号
    # !"#$%&\'()+,-./:;<=>?@[\\\]^_`{|}~
    # ！“”#￥%&、‘’（）+，-。/：；《=》？@【\\\】…—·{|}~
    pattern = (r'[\u4e00-\u9fff\u3000-\u303f\uac00-\ud7af'
               r'\u0020-\u007e\u00a0-\u00ff'
               r'!"#$%&\'()+,\-./:;<=>?@[\\\]^_`{|}~'
               r'！“”￥、‘’（），。：；《》？【\\\】…—·]+')
    return ''.join(re.findall(pattern, text))


def is_chinese(text):
    # 中文
    pattern = r'[\u4e00-\u9fff\u3000-\u303f\uac00-\ud7af]+'
    if len(re.findall(pattern, text)) > 0:
        return True
    else:
        return False


def is_english(text):
    # 英文
    pattern = r'[a-zA-Z]+'
    if len(re.findall(pattern, text)) > 0:
        return True
    else:
        return False


import shutil
import subprocess


def fix_mci(audio_path, output_path=None):
    if os.path.exists(output_path):
        return output_path
    if output_path:
        subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-i', audio_path, '-y', output_path])
        audio_path = output_path
    else:
        base_dir = os.path.dirname(audio_path)
        name, suffix = os.path.splitext(os.path.basename(audio_path))
        dst_path = os.path.join(base_dir, f'{name}_tmp{suffix}')
        shutil.copy(audio_path, dst_path)
        subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-i', dst_path, '-q', '-y', audio_path])
        os.remove(dst_path)
    return audio_path


def loud_normalize(out, loud):
    """
    使用EBU R128标准校正音频
    :param out:
    :param loud:
    """
    dir_name = os.path.dirname(out)
    name, suffix = os.path.splitext(os.path.basename(out))
    out_path = os.path.join(dir_name, f'{name}_{loud}{suffix}')
    cmd = ['ffmpeg', '-v', 'quiet', '-y', '-i', out, '-af', f'loudnorm=I={loud}:TP=-2:LRA=11', '-c:a', 'pcm_s16le', out_path]
    import subprocess
    subprocess.call(cmd)
    return out_path


import time
from contextlib import contextmanager


@contextmanager
def timer(tag_name: str):
    start = time.time()
    yield
    print(f'[{tag_name}] done in {time.time() - start:.2f}s @{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')


def detect_db(audio_path):
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    from math import log10
    audio = AudioSegment.from_file(audio_path, "wav")
    chunks = make_chunks(audio, 500)
    total_energy = 0
    max_chunk_energy = 0
    min_chunk_energy = sys.maxsize
    for chunk in chunks:
        chunk_energy = sum(map(abs, chunk.raw_data))
        total_energy += chunk_energy
        max_chunk_energy = max(max_chunk_energy, chunk_energy)
        min_chunk_energy = min(min_chunk_energy, chunk_energy)
    average_energy = total_energy / len(chunks)
    # max, mean, min
    return round(log10(max_chunk_energy), 2), round(log10(average_energy), 2), round(log10(min_chunk_energy), 2)


def detect_silence(audio_path, min_len=100, thresh=-50):
    from pydub import AudioSegment, silence
    audio = AudioSegment.from_file(audio_path, "wav")
    silence_list = silence.detect_silence(audio, min_silence_len=min_len, silence_thresh=thresh, seek_step=1)
    silence_list = sorted(set([item for sub_list in silence_list for item in sub_list]))
    return silence_list


def detect_eletric(audio_path):
    import librosa
    import numpy as np
    y, sr = librosa.load(audio_path, sr=None)
    mag, _ = librosa.magphase(librosa.stft(y))
    mag_dB = librosa.amplitude_to_db(mag, ref=np.max)

    n_fft = 2048
    freq_limit = 20000  # 假设我们关注20kHz以上的频率
    bin_index = int(np.round(freq_limit / (sr / n_fft)))
    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_freq_indices = np.where(fft_frequencies > bin_index)[0]
    low_freq_indices = np.where(fft_frequencies <= bin_index)[0]

    high_freq_energy = np.mean(mag_dB[high_freq_indices, :])
    mean_freq_energy = np.mean(mag_dB[:, :])
    low_freq_energy = np.mean(mag_dB[low_freq_indices, :])
    # high, mean, low
    return round(high_freq_energy, 2), round(mean_freq_energy, 2), round(low_freq_energy, 2)


def detect_asr(audio_dir):
    asr_file = "E:\\ET_TTS\\outputs_v2\\seed\\seed.list"
    if not os.path.exists(asr_file):
        from GPT_SoVITS.tools.asr.fasterwhisper_asr import execute_asr
        output_path = execute_asr(input_folder=audio_dir, output_folder=audio_dir, model_size='large-v3',
                                  language='en', precision='float16', extens=['.wav'])
        print(output_path)
    list_asr = []
    with open(asr_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == '': continue
            file_path, file_name, language, asr_text = line.split('|')
            list_asr.append({
                'file_path': file_path,
                'file_name': file_name,
                'language': language,
                'asr_text': asr_text,
            })
    # 计算asr相似度
    text = ('So, it’s a whole spectrum of collage, covering all bases, for your hair loss prevention,'
            'maintaining your nail health, keeping wrinkles at bay, or reducing sore muscles after exercise.')
    max_ratio = 0
    min_ratio = sys.maxsize
    import difflib
    for item in list_asr:
        ratio = difflib.SequenceMatcher(None, text, item['asr_text']).ratio()
        max_ratio = max(max_ratio, ratio)
        min_ratio = min(min_ratio, ratio)
        print('ratio==>', ratio, item['file_path'])
    # 结束算法
    print('max_ratio=', max_ratio, 'min_ratio=', min_ratio)


def detect_pesq(audio_file, ref_file, fs=16000):
    from scipy.io import wavfile
    from pesq import pesq
    import librosa

    # rate, ref = wavfile.read(ref_file)
    ref, sr = librosa.load(ref_file, sr=fs)
    # rate, deg = wavfile.read(audio_file)
    deg, sr = librosa.load(audio_file, sr=fs)

    nb_pesq = pesq(fs, ref, deg, 'nb')
    wb_pesq = pesq(fs, ref, deg, 'wb')
    print(audio_file, '===>', round(nb_pesq, 2), round(wb_pesq, 2))
    return nb_pesq, wb_pesq


if __name__ == '__main__':
    from et_dirs import outputs_v2
    base_dir = os.path.join(outputs_v2, f'seed')
    # asr检查
    # detect_asr(base_dir)
    # 特征检查
    audio_list = os.listdir(base_dir)
    audio_list = [file_name for file_name in audio_list if file_name.endswith('.wav')]
    audio_list = [os.path.join(base_dir, f'{name}') for name in audio_list]
    # max_high = 0
    # min_high = sys.maxsize
    # max_low = 0
    # min_low = sys.maxsize
    # from et_dirs import resources
    # ref_audio = os.path.join(resources, f'example_reference.mp3')
    # for path in audio_list:
        # print(path, '===>', detect_silence(path))
        # high, mean, low = detect_db(path)
        # high, mean, low = detect_eletric(path)
        # print(path, '===>', high, mean, low)
        # max_high = max(max_high, abs(high))
        # min_high = min(min_high, abs(high))
        # max_low = max(max_low, abs(low))
        # min_low = min(min_low, abs(low))
        # detect_pesq(path, ref_audio)
    # print('max/min===>', max_high, min_high, max_low, min_low)
