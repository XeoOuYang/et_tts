import os.path


def sentence_split(text, min_len=10):
    # 分割：文本，标点
    import re
    item_list = re.split('([.!?\n])', text)
    item_list = [sent.strip() for sent in item_list if sent != '']
    # 组装：文本+标点
    if len(item_list) % 2 == 1: item_list.append('.')   # 兼容无符号结尾文本
    concat_item = [''.join(group) for group in zip(item_list[0::2], item_list[1::2])]
    # todo: 合并短句子
    return concat_item

if __name__ == '__main__':
    text = ("¡Hola amigos, bienvenidos de regreso! "
            "Hoy es todo sobre sentirse increíble desde dentro hacia fuera, y tengo el producto perfecto para ti: Peptidos de Colágeno Multi de Micro Ingredientes y Café de Hongos. "
            "Confía en mí, estos dos se convertirán en tus nuevos mejores amigos para una vida más saludable y feliz! "
            "¡Así que prepárate para sentirte increíble! "
            "¡Vamos a empezar! "
            "¡Estoy emocionado de compartir con vosotros estos dos productos revolucionarios! "
            "¡Vamos a ver cómo pueden cambiar tu vida!")
    sentences = sentence_split(text)
    print(sentences, len(sentences))
    from coqui_ai import CoquiTTS
    tts = CoquiTTS(language='Spanish')
    # pre-load
    from et_dirs import resources
    ref_spk = os.path.join(resources, 'ref_spanish_59s.wav')
    tts.tts("¡Hola amigos, bienvenidos de regreso! ", ref_spk, language="Spanish")
    # test
    output = os.path.join(tts.output_dir, 'seq_run_sp.wav')
    from et_base import timer
    # 一次
    with timer('seq_run'):
        ret = tts.tts(text, ref_spk, output=output, language="Spanish")
        print(ret)
    # 多次
    with timer('batch_run'):
        for idx, sent in enumerate(sentences):
            output = os.path.join(tts.output_dir, f'batch_run_sp_{idx}.wav')
            with timer(f'batch_{idx}'):
                ret = tts.tts(sent, ref_spk, output=output, language="Spanish")
                print(ret)
    # 测试
