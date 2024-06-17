import re
import numpy as np
import wave

from num2words import num2words


def text_split(text, language) -> list[str]:
    text = text.strip("\n")
    punctuation_mark = r'[.?!:。？！：]'
    item_list = re.split(f'({punctuation_mark})', text)
    merge_item = [''.join(group) for group in zip(item_list[::2], item_list[1::2])]
    if len(item_list) % 2 == 1:
        merge_item.append(item_list[-1])
    # 长度对齐，并行加速
    # print('merge_item=>', merge_item)
    expt_length = 80 if language == 'english' else 40
    merge_item = length_align([i.lower().strip() for i in merge_item], expt_length=expt_length, split=True)
    # print('merge_item=>', merge_item)
    return merge_item


def length_align(merge_item, expt_length=120, split=True):
    align_list = []
    cur_text = ''
    for idx, text in enumerate(merge_item):
        if len(cur_text) + len(text) > expt_length:
            if cur_text != '':
                align_list.append(cur_text)
                cur_text = text
            else:
                if split:
                    align_list.extend(sentence_split(text, expt_length))
                else:
                    align_list.append(text)
                cur_text = ''
        else:
            cur_text += text
    # 最后一句处理
    if cur_text != '':
        if len(cur_text) < expt_length/2 and len(align_list) > 0:
            align_list[-1] += cur_text
        else:
            align_list.append(cur_text)
    return align_list


def sentence_split(text, expt_length) -> list[str]:
    _len_ = len(text)
    if _len_ < expt_length:
        return [text]
    # comma分句
    comma_mark = r'[,，]'
    item_list = re.split(f'({comma_mark})', text)
    merge_item = [''.join(group) for group in zip(item_list[::2], item_list[1::2])]
    if len(item_list) % 2 == 1:
        merge_item.append(item_list[-1])
    merge_item = [i.strip() for i in merge_item]
    return length_align(merge_item, expt_length, split=False)


def remove_punctuation(text):
    punctuation_pattern = r"[：；（）【】『』「」《》－‘“’”:;\(\)\[\]><\-'\"]"
    text = re.sub(punctuation_pattern, ' ', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'。{2,}', '。', text)
    return text


def r_strip(text: str, ref: str):
    last_start = text.rfind(']')
    if len(ref) < last_start < len(text):
        last_text = text[last_start+1:]
        word_list = last_text.split()
        word_list.reverse()
        for idx, word in enumerate(word_list):
            if word.strip() not in ref: word_list[idx] = ''
            else: break
        # 插入空格符
        word_list.append(' ')
        word_list.reverse()
        # 返回删除后格式
        return text[:last_start+1] + ' '.join(word_list)
    # 返回原型
    return text


def normalize_infer_text(text, ref, lang='english'):
    if lang == 'english':
        # 中文汉字
        adjust_pattern = re.compile(r'\b[a-zA-Z]*[\u4e00-\u9fa5]+[a-zA-Z]*\b')
        text = re.sub(adjust_pattern, '', text)
        # bad case
        # text = re.sub(r'](.*?)\s+california\b', '] ', text)
        # text = re.sub(r']\w+\b', '] ', text)
        # text = re.sub(r']\s+(tan|io|so|p)\b', '] ', text)
        # text = re.sub(r']\s+like\s*(p|io)?\b', '] ', text)
        # 新方法
        if len(ref) > 0: text = r_strip(text, ref)
    elif lang == 'chinese':
        text = re.sub(r'（.*?）', '', text)
        # text = re.sub(r'\b[\u4e00-\u9fa5]*[a-zA-Z]+[\u4e00-\u9fa5]*\b', '', text)
    # 连续空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def replace_dollar_sign(text):
    pattern = re.compile(r'(\$((\d+)(\.\d{2})?))')
    matches = pattern.findall(text)
    if matches:
        for match in sorted(matches, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(match[0], f'{num2words(match[1], lang="en")} dollars')
    text = text.replace('$', 'dollar')
    return text


def replace_percentage_sign(text):
    pattern = re.compile(r'(((\d+)(\.\d{2})?)%)')
    matches = pattern.findall(text)
    if matches:
        for match in sorted(matches, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(match[0], f'{num2words(match[1], lang="en")} percentages')
    text = text.replace('%', 'percentage')
    return text


def replace_numeric(text):
    pattern = re.compile(r'((\d+)(\.\d{2})?)')
    matches = pattern.findall(text)
    if matches:
        for match in sorted(matches, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(match[0], f'{num2words(match[0], lang="en")} ')
    text = re.sub(r'[\b]+', ' ', text)
    return text


character_map = {
    "：": "，",
    "；": "，",
    "！": "。",
    "（": "，",
    "）": "，",
    "【": "，",
    "】": "，",
    "『": "，",
    "』": "，",
    "「": "，",
    "」": "，",
    "《": "，",
    "》": "，",
    "－": "，",
    "‘": " ",
    "“": " ",
    "’": " ",
    "”": " ",
    '"': " ",
    "'": " ",
    ":": ",",
    ";": ",",
    "!": ".",
    "(": ",",
    ")": ",",
    "[": ",",
    "]": ",",
    ">": ",",
    "<": ",",
    "-": ",",
    "~": " ",
    "～": " ",
    "/": " ",
}

character_to_word = {
    " & ": " and ",
}


def apply_character_to_word(text):
    for k, v in character_to_word.items():
        text = text.replace(k, v)
    return text


def apply_character_map(text):
    translation_table = str.maketrans(character_map)
    return text.translate(translation_table)


def text_normalize(text, is_tts):
    # 删除括号
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'（.*?）', '', text)
    # 删除标点
    text = remove_punctuation(text)
    # 删除$, %
    text = replace_dollar_sign(text)
    text = replace_percentage_sign(text)
    # 数字转英文
    text = replace_numeric(text)
    # if is_tts: text = insert_spaces_between_uppercase(text)
    # 删除连续空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def insert_spaces_between_uppercase(s):
    # 使用正则表达式在每个相邻的大写字母之间插入空格
    return re.sub(
        r"(?<=[A-Z])(?=[A-Z])|(?<=[a-z])(?=[A-Z])|(?<=[\u4e00-\u9fa5])(?=[A-Z])|(?<=[A-Z])(?=[\u4e00-\u9fa5])",
        " ", s, )


def replace_unk_tokens(text, vocab):
    """
    把不在字典里的字符替换为 " , "
    vocab = chat_tts.pretrain_models["tokenizer"].get_vocab()
    """
    vocab_set = set(vocab.keys())
    # 添加所有英语字符
    vocab_set.update(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    vocab_set.update(set(" \n\r\t"))
    replaced_chars = [char if char in vocab_set else " , " for char in text]
    output_text = "".join(replaced_chars)
    return output_text


def batch_split(items, batch_size=5):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def combine_audio(wav_arr):
    wav_arr = [normalize_audio(w) for w in wav_arr]  # 先对每段音频归一化
    combined_audio = np.concatenate(wav_arr, axis=1)  # 沿着时间轴合并
    return normalize_audio(combined_audio)  # 合并后再次归一化


def normalize_audio(audio):
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def save_audio(file_name, audio, rate=24000):
    audio = (audio * 32767).astype(np.int16)
    with wave.open(file_name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


if __name__ == '__main__':
    # print(text_normalize("You can still enjoy a fantastic deal with the 1lb bag at $16 and the 2lb bag at $24.94."))
    # print(text_normalize("$64.94"))
    # print(normalize_infer_text("阻ing the effects of [uv_break] like aging?"))

    # text = "In terms of the collagen, guys, the reason this collagen supplement is so popular, if you've ever taken collagen supplements before. "
    # print(batch_split(text_split(text_normalize(text))))

    text = "it's just like the all natural, [uv_break] like non gmo solution [uv_break] for women who just want to look and feel their best [uv_break] like [uv_break] like [uv_break] io"
    print(normalize_infer_text(text, text))
