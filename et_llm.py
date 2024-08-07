import re

from LLM_AI.llm_llama import LLM_Llama_V3
from LLM_AI.llm_glm import LLM_GLM_4

LLM_INSTANCE = {
    'llama_v3': LLM_Llama_V3(),
    # 'llama_v3': LLM_Llama_V3(model_name='Mecord-FT/Meta-Llama-3-8B-V10'),
    # 'llama_v3': LLM_Llama_V3(model_name='FlagAlpha/Llama3-Chinese-8B-Instruct'),
    # 'llama_v3': LLM_Llama_V3(model_name='ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3'),
    # 'llama_v3': LLM_Llama_V3(model_name='LLM-Research/Llama3-8B-Chinese-Chat'),
    'glm_4': LLM_GLM_4()
}


def is_load(model: str):
    return model in LLM_INSTANCE and LLM_INSTANCE[model].is_load()


def unload_model(model: str):
    assert model in LLM_INSTANCE
    LLM_INSTANCE[model].unload_model()
    LLM_VERSION[model] = False


LLM_VERSION = {key: is_load(key) for key in LLM_INSTANCE.keys()}


def count_sentence(text):
    pattern = r'[．?!:。？！：]|\.'
    matches = re.findall(pattern, text)
    count = len(matches)
    return count


def llm_glm_4(query, role_play, context, inst_text, **kwargs):
    if is_load('llama_v3'):
        unload_model('llama_v3')
    # 加载新模型
    llm = LLM_INSTANCE['glm_4']
    LLM_VERSION['glm_4'] = True
    system_text = f'{role_play}' if context == '' else f'{role_play}\n{context}'
    query = f'{inst_text}' if query == '' else f'{inst_text}\n{query}'
    an = llm.llm(query=query, system=system_text, **kwargs)
    # 对内容格式化
    max_num_sentence = 0 if 'max_num_sentence' not in kwargs else kwargs['max_num_sentence']
    return _post_llm_(an, max_num_sentence)


def llm_llama_v3(query, role_play, context, inst_text, **kwargs):
    if is_load('glm_4'):
        unload_model('glm_4')
    # 加载新模型
    llm = LLM_INSTANCE['llama_v3']
    LLM_VERSION['llama_v3'] = True
    system_text = f'{role_play}' if context == '' else f'{role_play}\n{context}'
    query = f'{inst_text}' if query == '' else f'{inst_text}\n{query}'
    an = llm.llm(query=query, system=system_text, **kwargs)
    # 对内容格式化
    max_num_sentence = 0 if 'max_num_sentence' not in kwargs else kwargs['max_num_sentence']
    return _post_llm_(an, max_num_sentence)


def llm(query, role_play, context, inst_text, **kwargs):
    """
    # query：输入llm大模型的文本，从prompt_utils获取对应指令
    # role_play：扮演角色定义
    # context：产品详情；运营期望回复指令说明样例
    # inst_text：系统指令，具体说明回复语言，格式等
    # max_num_sentence：最大输出句子数量，从prompt_utils获取对应指令时返回预设数量
    """
    # an = llm_llama_v3(query, role_play, context, inst_text, **kwargs)
    an = llm_glm_4(query, role_play, context, inst_text, **kwargs)
    # 对内容格式化
    max_num_sentence = 0 if 'max_num_sentence' not in kwargs else kwargs['max_num_sentence']
    return _post_llm_(an, max_num_sentence)


def _post_llm_(an, max_num_sentence):
    # 过滤不可打印字符
    from et_base import reserve_char_all
    an = reserve_char_all(an)
    # 过滤不符合规范字符
    if an.startswith('.'): an = an[1:]
    # # 断句限制
    # if max_num_sentence > 0:
    #     arr_list = re.split(r'\.|\?|!|。|？|！', an)
    #     if len(arr_list) > max_num_sentence:
    #         arr_list[max_num_sentence] = ''
    #         an = '.'.join(arr_list[:max_num_sentence + 1])
    # 删除空白括号
    an = re.sub(r'\(\s*\)', '', an)
    an = re.sub(r'（\s*）', '', an)
    # 输出文本格式
    an = re.sub(r'(\.)\1+', '', an)
    an = re.sub(r'(_)\1+', '', an)
    an = re.sub(r'\s+', ' ', an)
    an = an.replace('*', '')
    return an


if __name__ == '__main__':
    query = 'is it really useful?'
    context = ('Viagra Erectile dysfunction can be treated with sildenafil (also called sexual impotence). '
               'Phosphodiesterase 5 (PDE5) inhibitors include Sildenafil. Anti-phosphodiesterase type-5 drugs are '
               'used to slow down an enzyme called phosphodiesterase type 5. This enzyme can be found in the penis.')
    from et_base import timer
    with timer('qa_llama_v3'):
        # an = llm_llama_v3(query, role_play='You are a super seller. You are selling products in air now. ',
        #                   context=context, inst_text='You can only rely in English.')
        an = llm_glm_4(query, role_play='You are a super seller. You are selling products in air now. ',
                          context=context, inst_text='You can only rely in English.')
    print(an)
    # todo: 只允许英文/汉字
