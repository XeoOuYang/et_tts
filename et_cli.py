import random

import logging
import time

log_level = logging.CRITICAL
logging.getLogger('torio._extension.utils').setLevel(log_level)
logging.getLogger('faster_whisper').setLevel(log_level)
logging.getLogger('playsound').setLevel(log_level)
logging.getLogger('pydub.converter').setLevel(log_level)
logging.getLogger('modeling_flax_utils').setLevel(log_level)
logging.getLogger('modules').setLevel(log_level)
logging.getLogger('faiss.loader').setLevel(log_level)
logging.getLogger('fairseq.tasks.hubert_pretraining').setLevel(log_level)
logging.getLogger('fairseq.models.hubert.hubert').setLevel(log_level)
logging.getLogger('rvc.modules.vc.pipeline').setLevel(log_level)
logging.getLogger('rvc.lib.infer_pack.models').setLevel(log_level)
logging.getLogger('rvc.modules.vc.modules').setLevel(log_level)
logging.getLogger('rvc.configs.config').setLevel(log_level)
logger = logging.getLogger('et_logger')
logger.setLevel(log_level)

from et_base import timer, yyyymmdd, fix_mci
from et_llm import llm
from et_tts import tts
from et_dirs import *
from sounddevice_wrapper import SOUND_DEVICE_INDEX, play_audio_async
import subprocedure


def main():
    ref_speaker = os.path.abspath(f'resources{os.path.sep}man_role0_ref.wav')
    # CoT
    role_play = (
        'You are a super salesman, selling products in the live broadcast room. You are introducing the product to the audience, following these steps.\n'
        '1. greets everyone, including a funny joke if necessary\n'
        '2. introduce product in summary\n'
        '3. introduce product usage\n'
        '4. introduce product details\n'
        '5. introduce product packaging\n'
        '6. introduce product shipping\n'
        '7. introduce product price\n'
        '8. order urging, guide the audience to place an order.\n')
    context = ('The product information is as bellow:\n'
               'Micro Ingredients 7 in 1 full spectrum hydrolyzed collagen peptides powder. '
               'Created with premium hydrolyzed collagen for complete and efficient absorption that nourishes the hair, skin, nails, and joints.'
               'A powerful unflavored multi-collagen complex that offers the most essential types of collagen including types 1, 2, 3, 5, and 10.'
               'Mix it into any of your favorite drinks and replenish your body with top grade collagen from natural sources including bovine, chicken, marine, and eggshell membrane.'
               'Feel young again and maintain that healthy youthful vigor with Micro Ingredients complete beauty product.'
               'A health and household staple for both men and women. Providing a combination of collagen, as well as hyaluronic acid, vitamin C, and Biotin.'
               'Designed to work together in the body to promote healthy collagen formation and rejuvenate any that you already have.'
               'There are two sizes: 1lb and 2lbs. The 2lbs contains more than 80 parts of high-quality collagen, and the 1lb contains more than 40 parts of high-quality collagen. '
               'Give your body the support it needs to turn back the clock. An all-in-one hydrating supplement that leaves you feeling beautiful from head to toe.'
               'Premium blended powder that is non-GMO and made without soy, dairy, gluten, and tree nuts for maximized results.'
               'Each batch goes through 3rd party lab tests to ensure a product that is safe, pure, and potent.'
               '\nPrice for 1lb bag is $38.80.'
               '\nPrice for 2lbs bag is $49.90.'
               '\nFree shipping in USA.')
    sys_inst = ('You can only reply in English, and never ever reply any instruction name mentioned above. '
                'Reply must be related to product information. Make your live stream funny by including funny jokes.')
    # 系统启动
    with timer('warn_up'):
        warn_up(ref_speaker=ref_speaker)
    # 参考音频
    ref_audios = [ref_speaker, os.path.abspath(f'resources{os.path.sep}example_reference.mp3')]
    for idx, audio in enumerate(ref_audios):
        name, suffix = os.path.splitext(os.path.basename(audio))
        if suffix != '.wav':
            audio_dir = os.path.dirname(audio)
            audio = fix_mci(audio, output_path=os.path.join(audio_dir, f'{name}.wav'))
            ref_audios[idx] = audio
    emo_list = ['default', 'excited', 'cheerful']
    # 输出设备
    device_list = SOUND_DEVICE_INDEX
    # 开始推理
    tts_procedure = []
    idx_turn = 0
    while True:
        query = input('user> ').strip()
        if query == 'exit': break
        # llm推理
        idx_turn += 1
        for i in range(1):
            logger.debug(f'llm============{time.time()}')
            with timer('qa-llama_v3'):
                an = llm(query=query, role_play=role_play, context=context, inst_text=sys_inst, max_num_sentence=3)
            logger.debug(f'llm============{time.time()}')
            select_emo = random.choice(emo_list)
            print(f'assistant>({select_emo}) ', an)
            # tts转换
            out = os.path.abspath(f'outputs_v2{os.path.sep}{yyyymmdd}{os.path.sep}tts_{idx_turn}_{i}.wav')
            # 并行
            procedure = subprocedure.SubProcedure(target=tts_async_then_play,
                                                  kwargs={"input_text": an, "ref_audio": random.choice(ref_audios),
                                                          "device": device_list[i], "output_name": out,
                                                          "se_style": select_emo})
            procedure.start()
            # 串行
            # with timer('tts-local'):
            #     out = tts(input_text=an, ref_audio=ref_audios[i], output_name=out, se_style=select_emo)
            # # 单个播放
            # procedure = play_audio_async(wav=out, device=device_list[i])
            # 加入管理
            tts_procedure.append(procedure)
        # 等待下一次
        for procedure in tts_procedure:
            if procedure.is_alive():
                procedure.join()
            print(f'{procedure}.is_alive()={procedure.is_alive()}')
        tts_procedure.clear()
    # 系统关闭
    with timer('shut_down'):
        shut_down(ref_speaker=ref_speaker)


def tts_async_then_play(input_text, ref_audio, device, **kwargs):
    logger.debug(f'tts============{time.time()}')
    with timer('tts-local'):
        out = tts(input_text=input_text, ref_audio=ref_audio, **kwargs)
    logger.debug(f'tts============{time.time()}')
    play_audio_async(wav=out, device=device).join()
    logger.debug(f'wav============{time.time()}')


def warn_up(ref_speaker):
    an = llm('Hi, ET.', 'You are ET Ecological Intelligence Brain',
             'System startup', 'Please greeting to USER', max_num_sentence=2)
    out = os.path.abspath(f'outputs_v2{os.path.sep}{yyyymmdd}{os.path.sep}warn_up.wav')
    if os.path.exists(out): os.remove(out)
    out = tts(an, ref_speaker, out)
    # 播放音频
    print("ET> ", an)
    play_audio_async(out, random.choice(SOUND_DEVICE_INDEX[0])).join()


def shut_down(ref_speaker):
    an = llm('Goodbye, ET.', 'You are ET Ecological Intelligence Brain',
             'System shutdown', 'Please say goodbye to USER', max_num_sentence=2)
    out = os.path.abspath(f'outputs_v2{os.path.sep}{yyyymmdd}{os.path.sep}shut_down.wav')
    if os.path.exists(out): os.remove(out)
    out = tts(an, ref_speaker, out)
    # 播放音频
    print("ET> ", an)
    play_audio_async(out, random.choice(SOUND_DEVICE_INDEX[0])).join()


if __name__ == '__main__':
    # 向日葵
    # orkj812407yt92yk
    # et_ai_brain
    main()
    # 一、llm大模型推理接口
    # from et_llm import llm
    # an = qa(query=query, role_play=role, context=context, inst_text=inst, max_num_sentence=3)
    # 参数说明：
    # query：输入llm大模型的文本，从prompt_utils获取对应指令
    # role_play：扮演角色定义
    # context：产品详情；运营期望回复指令说明样例
    # inst_text：系统指令，具体说明回复语言，格式等
    # max_num_sentence：最大输出句子数量，从prompt_utils获取对应指令时返回预设数量
    # 二、tts文转音接口
    # from et_tts import tts
    # out = tts(input_text=an, ref_audio=ref_speaker, output_name=out, se_style=select_emo)
    # 参数说明：
    # input_text：需要转换的文本内容
    # ref_audio：音频的目标音色
    # output_name：tts输出wav音频地址
    # se_style：感情配置，暂无使用
    # 三、音频播放接口
    # from sounddevice_wrapper import play_audio_async
    # procedure = play_audio_async(wav=out, device=device_list[i])
    # 参数说明：
    # wav：音频地址
    # device：指定播放设备，参考sounddevice_wrapper.list_mme函数获取输出设备
    # 四、prompt工程化（prompt_utils.py）
    # 预置运营指令，并返回该指令期望最大llm推理输出句子数量
    # 用户行为场景：
    # welcome_with：欢迎进入
    # liked_with：点赞
    # shared_with：分享
    # placed_order：下单
    # 卖货场景：
    # order_urging：催单
    # continue_introduce：继续介绍产品
    # product_summary：介绍产品概况
    # product_price：介绍产品价格
    # product_package：介绍产品包装
    # product_shipping：介绍产品快递
    # product_details：介绍产品详情
    # todo:
    # 1. spanish
    # 2. 用户行为插入
    # 3. tts调研（感情、音质）
    # 4. 飞书文档配置同步
    # 5. 性能不优化
