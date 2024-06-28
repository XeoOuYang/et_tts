import json
import os.path
import random
import re
import shutil

import librosa
import numpy
import numpy as np
import pyAudioAnalysis.audioBasicIO

import torch
import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

from LLM_AI.llm_base import ForceTokenFixValueLogitsProcessor

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from ChatTTS.core import Chat
from et_base import ET_TTS


class SeedContext:
    @staticmethod
    def deterministic(seed=0, cudnn_deterministic=False):
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def __init__(self, seed: int, cudnn_deterministic=False):
        self.seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
        self.cudnn_deterministic = cudnn_deterministic
        self.state = None

    def __enter__(self):
        self.state = (
            torch.get_rng_state(),
            random.getstate(),
            numpy.random.get_state(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        try:
            SeedContext.deterministic(self.seed, cudnn_deterministic=self.cudnn_deterministic)
        except Exception as e:
            print(e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.state[0])
        random.setstate(self.state[1])
        numpy.random.set_state(self.state[2])
        torch.backends.cudnn.deterministic = self.state[3]
        torch.backends.cudnn.benchmark = self.state[4]


def clear_cuda_cache():
    torch.cuda.empty_cache()


class BadWordsLogitsProcessor(LogitsProcessor):
    def __init__(self, bad_words_token_id_list, special_token_id_list):
        self._bad_words_token_id_list = bad_words_token_id_list
        self._special_token_id_list = special_token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        processed_scores = scores.clone()
        processed_scores[:, self._bad_words_token_id_list] = -float('inf')
        return processed_scores


class ChatTTS(ET_TTS):
    def __init__(self, manual_seed: int = 5656, refine_text=True):
        super().__init__()
        self.model = Chat()
        from et_dirs import chattts_base, model_dir_base
        self.model_dir = os.path.join(os.path.join(model_dir_base, os.path.basename(chattts_base)), f'models')
        self.model.load_models(source='local', local_path=self.model_dir)
        self.samplerate = 24_000
        # 固定音色
        self.manual_seed = manual_seed
        self.skip_refine_text = not refine_text
        self.cached_spk_emb = {}
        # 中英文处理
        self.tokenizer = self.model.pretrain_models["tokenizer"]
        def is_special_token(token):
            if re.match(r'\[.*?\]', token):
                return True
            else:
                return False
        # 中文
        from et_base import is_chinese
        self._masked_indicator_cn = [token_id for token, token_id in self.tokenizer.vocab.items()
                                     if is_chinese(token) and not is_special_token(token)]
        # 英文
        from et_base import is_english
        self._masked_indicator_en = [token_id for token, token_id in self.tokenizer.vocab.items()
                                     if is_english(token) and not is_special_token(token)]
        # 特殊符号
        self._masked_indicator_special = [token_id for token, token_id in self.tokenizer.vocab.items()
                                          if is_special_token(token)]
        # like, california
        bad_words_list = ['like', 'california', 'io']
        def is_bad_word(text):
            for word in bad_words_list:
                if word in text: return True
            return False
        bad_words_token_id_list = [token_id for token, token_id in self.tokenizer.vocab.items()
                                   if is_bad_word(token) and not is_special_token(token)]
        self.bad_word_logits_processor = BadWordsLogitsProcessor(bad_words_token_id_list, self._masked_indicator_special)

    def sample_speaker(self, manual_seed):
        from ChatTTS.spec_voices.load_voice import spec_voice
        emb = spec_voice.get_by(manual_seed)
        # 有预设，直接返回
        if emb is not None: return emb
        # 否则, 随机生成
        with SeedContext(manual_seed, True):
            return self.model.sample_random_speaker()

    def tts(self, text: str, ref_speaker: str, **kwargs):
        # 固定音色
        manual_seed = kwargs['manual_seed'] if 'manual_seed' in kwargs else self.manual_seed
        if manual_seed not in self.cached_spk_emb:
            self.cached_spk_emb[manual_seed] = self.sample_speaker(manual_seed)
        use_spk_emb = self.cached_spk_emb[manual_seed]
        # use oral_(0-9), laugh_(0-2), break_(0-7)
        # oral: 插入口头禅(oral)强度，越大越多口头禅
        # laugh: 插入笑声(laugh)强度，越大越多笑声
        # break: 插入停顿(break)强度，越大越多停顿
        params_refine_text = {
            'prompt': '[oral_2][laugh_0][break_4]',
            'temperature': 0.3,           # 语言多样性
            'top_P': 0.7,                 # 语言自然度
            'top_K': 20,                  # 语言连贯性
            'repetition_penalty': 1.2,    # 重复吐词惩罚
            # 'max_new_token': 384,
        }
        if 'refine_prompt' in kwargs:
            params_refine_text['prompt'] = kwargs['refine_prompt']
        # 是否推理语气
        skip_refine_text = self.skip_refine_text
        if 'skip_refine_text' in kwargs:
            skip_refine_text = kwargs['skip_refine_text']
        # use speed_(0-9)
        # speed: 吐词速度
        params_infer_code = {
            'prompt': '[speed_5]',
            'spk_emb': use_spk_emb,       # speaker_id
            'temperature': 0.3,           # 语言多样性
            'top_P': 0.7,                 # 语言自然度
            'top_K': 20,                  # 语言连贯性
            # ‘repetition_penalty’: 1.05,
            # ‘max_new_token’: 2048,
        }
        if 'infer_prompt' in kwargs:
            params_infer_code['prompt'] = kwargs['infer_prompt']
        # 指定语言
        language = 'english' if 'language' not in kwargs else kwargs['language']
        language = language.lower()
        logits_processor = None
        if language == 'chinese':
            logits_processor = ForceTokenFixValueLogitsProcessor(self._masked_indicator_en)
        elif language == 'english':
            logits_processor = ForceTokenFixValueLogitsProcessor(self._masked_indicator_cn)
        print('language =', language)
        # 并行推理
        wav_list = []
        from ChatTTS.tools import text_normalize, text_split, batch_split

        def post_infer_text(_old, _new):
            """
            处理推理结果
            """
            import difflib
            re_new = re.sub(r'\[.*?\]', '', _new)
            if language == 'chinese':
                re_new = re.sub(r'\s+', '', re_new)
            ratio = difflib.SequenceMatcher(None, _old, re_new).ratio()
            print('input====>', _old)
            print('infer<====', _new, round(ratio, 2))
            # 判断ratio返回结果
            from ChatTTS.tools import normalize_infer_text
            if ratio > 0.5:
                return normalize_infer_text(_new, _old, language)
            else:
                return _old
        # 降低[uv_break]后like等词汇出现概率
        logits_processor_list = LogitsProcessorList([self.bad_word_logits_processor])
        if logits_processor is not None:
            logits_processor_list.append(logits_processor)
        with SeedContext(manual_seed, True):
            for batch in batch_split(text_split(text_normalize(text, True, language), language)):
                wav_arr = self.model.infer(batch, skip_refine_text=skip_refine_text, params_refine_text=params_refine_text,
                                           normalize_infer_text=post_infer_text, params_infer_code=params_infer_code, use_decoder=True,
                                           extra_refine_logits=logits_processor_list)
                wav_list.extend(wav_arr)
                clear_cuda_cache()
        wav_path = f'{self.output_dir}{os.path.sep}tmp.wav'
        from ChatTTS.tools import combine_audio, save_audio
        audio = combine_audio(wav_list)
        save_audio(wav_path, audio, self.samplerate)
        # 串行推理
        # wav_list = self.model.infer(text, params_infer_code=params_infer_code, use_decoder=True,
        #                             params_refine_text=params_refine_text, skip_refine_text=skip_refine_text)
        # clear_cuda_cache()
        # wav_path = f'{self.output_dir}{os.path.sep}tmp.wav'
        # from scipy.io import wavfile
        # wavfile.write(wav_path, self.samplerate, wav_list[0].T)
        # return result
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(wav_path, kwargs['output'])):
            shutil.copyfile(wav_path, kwargs['output'])
            os.remove(wav_path)
            wav_path = kwargs['output']
        return wav_path

def run_random_seed(text, start=0, end=10000, bias=0, ref_audio=''):
    # 读取缓存
    tts = ChatTTS(manual_seed=2000, refine_text=False)
    seed_base = os.path.join(outputs_v2, f'seed')
    if not os.path.exists(seed_base):
        os.makedirs(seed_base)
    seed_set_file = os.path.join(seed_base, 'seed.json')

    def load_seed_set(json_file) -> dict[int, str]:
        if not os.path.exists(json_file):
            return {}
        with open(json_file, 'r', encoding='utf8', errors='ignore') as fd:
            json_dict = json.loads(fd.read())
        return json_dict
    seed_dict = load_seed_set(seed_set_file)
    # 随机测试
    for _ in tqdm.tqdm(range(end)):
        manual_seed = random.randint(0, 100000) + bias
        if manual_seed not in seed_dict:
            prompt = f'[oral_2][laugh_0][break_4]'
            out_path = os.path.join(seed_base, f'chattts_{manual_seed}.wav')
            with timer('chat_tts'):
                out_path = tts.tts(text=text, ref_speaker=f'{manual_seed}', output=out_path,
                                   refine_prompt=prompt, manual_seed=manual_seed, language='chinese')
            seed_dict[manual_seed] = {'out_path': out_path}
        # 计算概率
        print(seed_dict[manual_seed]['out_path'])
        _f0_, _mfcc_, _dtw_mfcc_ = similarity(seed_dict[manual_seed]['out_path'], ref_audio)
        seed_dict[manual_seed]['f0'] = _f0_
        seed_dict[manual_seed]['mfcc'] = _mfcc_
        seed_dict[manual_seed]['dtw_mfcc'] = _dtw_mfcc_
        # _all_ = similarity_all(seed_dict[manual_seed]['out_path'], ref_audio)
        # seed_dict[manual_seed]['all'] = _all_
        _score_ = audio_similarity(seed_dict[manual_seed]['out_path'], ref_audio)
        seed_dict[manual_seed]['score'] = _score_
        print(seed_dict[manual_seed])
    # 保存结果
    with open(seed_set_file, 'w', encoding='utf8', errors='ignore') as fd:
        fd.write(json.dumps(seed_dict))


def mfcc(audio_path, sr=16000):
    audio, _sr = librosa.load(audio_path)
    if _sr != sr: audio = librosa.resample(y=audio, orig_sr=_sr, target_sr=sr)
    # 预加权
    preemph = 0.95
    audio = np.append(audio[0], audio[1:]-preemph*audio[:-1])
    from librosa.feature import mfcc
    # 计算mfcc特征，并且归一化
    audio_mfcc = mfcc(y=audio, sr=sr, n_mfcc=64)

    return np.mean(np.abs(audio_mfcc), axis=1), audio_mfcc

def f0(audio_path, sr=16000):
    """音高"""
    audio, _sr = librosa.load(audio_path)
    if _sr != sr: audio = librosa.resample(y=audio, orig_sr=_sr, target_sr=sr)
    from librosa import pyin, yin, note_to_hz
    audio_f0 = yin(y=audio, fmin=note_to_hz('C0'), fmax=note_to_hz('C7'), sr=sr)

    return np.mean(audio_f0)

def similarity_all(audio_path, ref_path):
    from pyAudioAnalysis import audioBasicIO
    from pyAudioAnalysis import ShortTermFeatures
    # 生成audio
    [Fs, x] = audioBasicIO.read_audio_file(audio_path)
    x = audioBasicIO.stereo_to_mono(x)
    F_audio, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    # 参考audio
    [Fs, x] = audioBasicIO.read_audio_file(ref_path)
    x = audioBasicIO.stereo_to_mono(x)
    F_ref, _ = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    # 特征计算
    F_audio = np.mean(F_audio, axis=1)
    F_ref = np.mean(F_ref, axis=1)
    from scipy.spatial.distance import cosine
    # _all_ = np.abs(F_audio-F_ref)/np.abs(F_ref)
    _all_ = cosine(F_audio, F_ref)
    return _all_

def similarity(audio_path, ref_path, sr=16000):
    """
    f0: 基频特征，说话越大声，值越大，响亮程度越高，负数就是说话比原来低沉，体现在音高上
    mfcc: 梅尔倒谱mean特征，使用曼哈顿距离计算，值越小，两者特征越相似，体现在说话频率上
    dtw_mfcc: dtw计算mfcc时序特征，表示音频整体相似程度，值越大与原音频越相似，体现在音色上
    """
    audio_mfcc, seq_audio_mfcc = mfcc(audio_path, sr)
    ref_mfcc, seq_ref_mfcc = mfcc(ref_path, sr)
    from scipy.spatial.distance import cityblock
    # 曼哈顿距离
    similarity_mfcc = cityblock(audio_mfcc, ref_mfcc)
    similarity_mfcc = similarity_mfcc/np.mean(ref_mfcc)
    # dtw计算距离
    global count_dtw
    count_dtw = 0
    from fastdtw import fastdtw
    def dist_dtw(x, y):
        global count_dtw
        count_dtw += 1
        return np.linalg.norm(x - y, ord=2)
    fastdtw_mfcc, _ = fastdtw(seq_audio_mfcc.T, seq_ref_mfcc.T, dist=dist_dtw)
    # f0特征
    audio_f0 = f0(audio_path, sr)
    ref_f0 = f0(ref_path, sr)
    similarity_f0 = (audio_f0-ref_f0)/ref_f0
    # 返回值
    return similarity_f0, similarity_mfcc, fastdtw_mfcc/count_dtw

def audio_similarity(audio_path, ref_path, sr=16000):
    """
    其他计算音频特征：
    zcr: 过零率
    rhythm: 语速节奏
    chroma: 表示两个音频点的差异程度
    energy: 振幅，体现在响度
    spectral: 能力谱特征， 越亮表示越大声
    perceptual: 能被人感知程度
    """
    from audio_similarity import AudioSimilarity
    weights = {
        'zcr_similarity': 0.2,
        'rhythm_similarity': 0.2,
        'chroma_similarity': 0.2,
        'energy_envelope_similarity': 0.1,
        'spectral_contrast_similarity': 0.1,
        'perceptual_similarity': 0.2
    }
    audio_similarity = AudioSimilarity(ref_path, audio_path, sr, weights, verbose=True)
    similarity_score = audio_similarity.stent_weighted_audio_similarity(metrics='all')
    return similarity_score

if __name__ == '__main__':
    from et_dirs import outputs_v2
    # text = ('So, it’s a whole spectrum of collage, covering all bases, for your hair loss prevention,'
    #         'maintaining your nail health, keeping wrinkles at bay, or reducing sore muscles after exercise.'
    #         'It covers everything!'
    #         'Not just limited to one specific part of ourselves but truly taking care of us as individuals.')
    # text = ('So, it’s a whole spectrum of collage, covering all bases, for your hair loss prevention,'
    #         'maintaining your nail health, keeping wrinkles at bay, or reducing sore muscles after exercise.')
    # text_list = [
    #     "Hello, gorgeous! Welcome to Natural Power! I'm so excited to be here with all of you tonight to talk about one of my absolute favorite topics: the power of natural ingredients to transform your beauty routine! I'm sipping on my daily dose of beauty right now, and let me tell you, this collagen elixir is my secret weapon for glowing skin, strong hair, and nails that wow! Today, we're diving deep into the world of collagen and unlocking the secrets to youthful, radiant beauty. And guess what? I've got an incredible deal for you today that you won't want to miss! Before we get started, drop a comment below and let me know where you're tuning in from! I'm curious to see how far and wide the Natural Power community reaches. Now, let's get this natural beauty party started!",
    #     "Ladies, are you ready to turn back the clock and unlock your youthful glow? I'm thrilled to introduce you to the Micro Ingredients 7-in-1 Full Spectrum Hydrolyzed Collagen Peptides Powder, your ultimate secret weapon for radiant skin, lustrous hair, strong nails, and supple joints! Are you feeling the effects of aging? Don't worry, darling! This collagen powder is here to rescue you from sagging skin, dull hair, brittle nails, and creaky joints. It's like a magic potion that nourishes your body from within, leaving you feeling and looking years younger! Okay, lovelies, let's talk about what makes this collagen powder an absolute game-changer. Trust me, it's not your average supplement! First of all, it's like a supercharged beauty smoothie for your body. We're talking a 7-in-1 full spectrum blend that delivers all the essential types of collagen your body craves. This means complete nourishment for your skin, hair, nails, joints – the whole package! And forget about those chunky, hard-to-digest collagen powders. This one is hydrolyzed, which means it's broken down into tiny particles for super-easy absorption. You'll get maximum benefits from every single scoop! Plus, it's made with all-natural goodness. No weird chemicals or artificial junk here. Just pure, clean collagen sourced from things like grass-fed cows, chickens, and even marine sources. It's like a little taste of nature for your body! Now, I know some of you might be thinking, ""But does it taste weird?"" Not at all! It's completely flavorless, so you can mix it into your morning coffee, your favorite smoothie, or even your water without changing the taste one bit. Oh, and did I mention it's packed with other beauty-boosting ingredients? We're talking hyaluronic acid for hydration, vitamin C for radiance, and biotin for strong hair and nails. It's like a multivitamin for your beauty routine!",
    #     "And the best part? This collagen powder isn't just for the ladies. Men, you can benefit from this too! It's the perfect way to keep your skin looking firm and your joints feeling young. We offer two sizes, so you can choose the one that fits your lifestyle. And if you're in the US, shipping is on us! So, what are you waiting for?",
    #     "And the best part? For our amazing new customers, we're offering an exclusive 50% discount! That means you can snag the 1lb bag for just $8.24 and the 2lb bag for only $12.47! And if you're already part of our fabulous community, don't worry, we've got you covered too!  You can still enjoy a fantastic deal with the 1lb bag at $16.47 and the 2lb bag at $24.94.",
    #     "Alright, my lovelies! It's time to take action and grab this incredible deal on our 7-in-1 Collagen Peptides Powder! First things first, make sure you click that little % sign in the top left corner of your screen. That's your golden ticket to an exclusive discount! Once you've claimed your coupon, simply tap the button below to check out the product and place your order. Trust me, your skin, hair, nails, and joints will thank you for it! Don't miss out on this amazing opportunity to boost your natural beauty from the inside out. Click that button now and let's get glowing!",
    #     "This is an absolute steal, ladies! Don't miss out on this incredible opportunity to transform your beauty routine without breaking the bank. Order your Micro Ingredients 7-in-1 Collagen Peptides Powder now and experience the magic for yourself! Don't miss out on this opportunity to reclaim your youthfulness. Invest in yourself and your beauty. Order your Micro Ingredients 7-in-1 Collagen Peptides Powder today and experience the transformation!",
    #     "Look at this elegant packaging! It's perfect for gifting or treating yourself. This one is good for your skin, nail, hair & joint. And let me tell you, this stuff is very easy to use. I just toss a scoop into my oatmeal every morning – boom, instant beauty breakfast! It's completely flavorless, so it won't mess with the taste of your favorite foods. You can mix it into anything you want – your coffee, your smoothies, even just plain water. It's like a secret weapon for getting gorgeous from the inside out. So, while you're enjoying your favorite drinks, your body is getting a major upgrade. Talk about a win-win situation!",
    #     'So hurry up！ Click the button below and order it！It will make you better！'
    # ]
    from et_base import timer
    # use oral_(0-9), laugh_(0-2), break_(0-7)[, speed_(0-9)]
    # oral: 插入词(oral)强度，越大越多口头禅
    # laugh: 笑声(laugh)强度，越大越多笑点
    # break: 停顿(break)强度，越大越多停顿
    # text = ('哥哥，你给我买这个棒棒糖，你女朋友不会生气吧？'
    #         '哎，真好吃，哥，你尝一口。'
    #         '哥哥，咱俩吃同一个棒棒糖，你女朋友知道了，不会吃醋吧？'
    #         '哥哥，你骑着小电动车带着我，你女朋友知道了不会揍我吧？'
    #         '你女朋友好可怕，不像我，只会心疼giegie。')
    text = ('回来得太晚了吧，你都睡着了。你怎么连睡着的样子都这么好看呀。可是你要是这么睡着的话，会感冒的。'
            '被子都不盖，你这个笨蛋在想什么呀。没事，没事，是我呀把你吵醒了。这样睡着的话，会感冒的，我帮你把被子盖上。'
            '睡吧，睡吧，对不起呀，我是不是回来得太晚了？我呀，我就看着你睡着之后的样子，在睡你都不知道你睡着的样子有多可爱！')
    from et_dirs import resources
    ref_audio = os.path.join(resources, '88795527.mp3')
    run_random_seed(text, 0, 500, bias=0, ref_audio=ref_audio)

    # with open('text.txt', 'r', encoding='utf8', errors='ignore') as fd:
    #     text = fd.read()
    # # 本地测试
    # tts = ChatTTS(manual_seed=410)
    # manual_seed = tts.manual_seed
    # # # 随机测试
    # for idx in range(1):
    #     prompt = f'[oral_2][laugh_0][break_4]'
    #     out_path = os.path.join(outputs_v2, f'chattts_{manual_seed}_{idx}.wav')
    #     with timer('chat_tts'):
    #         out_path = tts.tts(text=text, ref_speaker=f'{manual_seed}', output=out_path,
    #                            refine_prompt=prompt, manual_seed=manual_seed)
    #     print(out_path)
