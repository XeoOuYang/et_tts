import re

from LLM_AI.llm_llama import LLM_Llama_V3
from LLM_AI.llm_glm import LLM_GLM_4

LLM_INSTANCE = {
    'llama_v3': LLM_Llama_V3(),
    # 'llama_v3': LLM_Llama_V3(model_name='Mecord-FT/Meta-Llama-3-8B-V6'),
    'glm_4': LLM_GLM_4()
}


def is_load(model: str):
    return model in LLM_INSTANCE and LLM_INSTANCE[model].is_load()


def unload_model(model: str):
    assert model in LLM_INSTANCE
    LLM_INSTANCE[model].unload_model()


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
    system_text = f'{role_play}\n\n{context}\n\n{inst_text}'
    an = llm.llm(query=query, system=system_text, **kwargs)
    # 对内容格式化
    max_num_sentence = 0 if 'max_num_sentence' not in kwargs else kwargs['max_num_sentence']
    return _post_llm_(an, max_num_sentence)


def llm_llama_v3(query, role_play, context, inst_text, **kwargs):
    if is_load('glm_4'):
        unload_model('glm_4')
    # 加载新模型
    llm = LLM_INSTANCE['llama_v3']
    system_text = f'{role_play}\n\n{context}\n\n{inst_text}'
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
    if an.startswith('.'): an = an[1:]
    # 如果有断句限制
    if max_num_sentence > 0:
        arr_list = re.split(r'\.|\?|!|。|？|！', an)
        if len(arr_list) > max_num_sentence:
            arr_list[max_num_sentence] = ''
            an = '.'.join(arr_list[:max_num_sentence + 1])
    # 对llm输出文本格式化
    an = re.sub(r'(\.)\1+', '.', an)
    an = re.sub(r'(_)\1+', '', an)
    an = re.sub(r'\s+', ' ', an)
    an = re.sub(r'[<>|=#&^\\/]', '', an)
    an = re.sub(r'\*.*?\*', '', an).replace('**', '')
    an = re.sub(r'\(.*?\)|\{.*?\}|\[.*?\]|<.*?>', '', an)
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
    # todo: 禁止罗马数字
    # todo: 只允许英文/汉字
    # todo: 静止非法字符
