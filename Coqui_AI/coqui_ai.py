import os.path
import shutil

import torch
import torchaudio

from et_base import ET_TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

SES_DICT = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
}

class CoquiTTS(ET_TTS):
    def __init__(self, language='English'):
        super().__init__()
        config = XttsConfig()
        from et_dirs import coqui_ai_base, model_dir_base
        model_dir = os.path.join(model_dir_base, os.path.basename(coqui_ai_base), 'models')
        config_path = os.path.join(model_dir, 'XTTS-v2', 'config.json')
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        ckpt_dir = os.path.join(model_dir, 'XTTS-v2')
        model.load_checkpoint(config, checkpoint_dir=ckpt_dir, eval=True)
        model.cuda()
        # 保持模型
        self.config = config
        self.model = model
        self.language = SES_DICT[language]
        # 缓存ref_speaker
        self.speakers = {}

    def tts(self, text: str, ref_speaker: str, **kwargs):
        language = self.language
        if 'language' in kwargs:
            language = SES_DICT[kwargs['language']]
        # 开始推理
        # v1版本只支持239个字符
        # wav_path = self.tts_v1(text, ref_speaker, language)
        # v2版本无限, 试验batch看看
        wav_path = self.tts_v2(text, ref_speaker, language)
        # 存储结果
        if 'output' in kwargs and (not os.path.exists(kwargs['output'])
                                   or not os.path.samefile(wav_path, kwargs['output'])):
            shutil.copyfile(wav_path, kwargs['output'])
            os.remove(wav_path)
            wav_path = kwargs['output']
        return wav_path

    def tts_v1(self, text, ref_speaker, language):
        # 开始推理
        # https://docs.coqui.ai/en/dev/models/xtts.html
        outputs = self.model.synthesize(text, self.config, speaker_wav=ref_speaker, language=language)
        wav_path = f'{self.output_dir}{os.path.sep}tmp.wav'
        torchaudio.save(wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
        # 返回结果
        return wav_path

    def tts_v2(self, text, ref_speaker, language):
        # 提取参考声纹
        from et_base import timer
        with timer('coqui_emb'):
            ITEM_KEY = str(abs(hash(ref_speaker)))
            if ITEM_KEY not in self.speakers:
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[ref_speaker])
                self.speakers[ITEM_KEY] = (gpt_cond_latent, speaker_embedding)
            else:
                gpt_cond_latent, speaker_embedding = self.speakers[ITEM_KEY]
        # 开始推理
        # https://docs.coqui.ai/en/dev/models/xtts.html
        outputs = self.model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7,
            # Add custom parameters here
            enable_text_splitting=True,
        )
        wav_path = f'{self.output_dir}{os.path.sep}tmp.wav'
        torchaudio.save(wav_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
        # 返回结果
        return wav_path

if __name__ == '__main__':
    tts = CoquiTTS(language='Spanish')
    from et_dirs import resources
    from et_base import timer
    # ref_speaker = os.path.join(resources, 'ref_spanish_59s.wav')
    # with timer('tts-es'):
    #     output = tts.tts('Si están intentando comprar ambos, presione el ícono del carrito de compras en la esquina y agréguelos ambos al carrito.'
    #                      'Sólo para resumir para aquellos de ustedes que son nuevos. ¡Bien! Colágeno, ideal para cabello, piel, uñas, articulaciones.'
    #                      'Hay cinco tipos de colágeno aquí. Biotina, vitamina C, ácido hialurónico. Entonces es un suplemento de colágeno de espectro completo, ¿verdad?'
    #                      'Cubre todas las bases allí. A diferencia de la mayoría de los suplementos de colágeno, la mayoría de los suplementos de colágeno solo tienen el tipo uno, ¿verdad?'
    #                      'Entonces este es el espectro completo.',
    #                      ref_speaker, language='Spanish')
    #     print(output)
    # '你好，我叫kate。我非常的fashion和international，我也会说一点日语，こんにちは。'
    # ref_speaker = os.path.join(resources, '88795527.mp3')
    ref_speaker = "C:\\Users\\1\\Downloads\\tts\\ttt.wav"
    # text = ('小兔子和小狐狸在大家眼中是一對奇怪的情侶，小兔子總是慢騰騰的，明天要交差的活，今天絕對不會幹完，'
    #         '平時不管做什麽事都磨磨蹭蹭的，是大家公認的慢性子。而小狐狸可是個妥妥的急性子，做事雷厲風行，'
    #         '卻常常是火急火燎的，總會因為太著急而出錯。就這樣兩個根本不在一條路上的人，居然在一起了。')
    # import zhconv
    # text = zhconv.convert(text, 'zh-cn')
    text = "안녕하세요! 오늘은 충전이 필요 없는 스마트 쓰레기통 UOCAN을 소개해드리겠습니다. 레드닷 디자인 어워드를 수상한 이 쓰레기통은 예술 작품처럼 세련된 디자인을 자랑합니다. UOCAN은 쓰레기 봉투를 자동으로 늘리고, 묶고, 자를 수 있어 봉투 사용량을 최대 70% 절약할 수 있습니다. 저희의 쓰레기 봉투는 매우 튼튼하고 찢어지지 않으며, 환경 친화적입니다. UOCAN, 더 스마트하고 우아한 쓰레기 처리를 경험하세요."
    with timer('tts-ml'):
        output = tts.tts(text , ref_speaker, language='Korean')
        print(output)