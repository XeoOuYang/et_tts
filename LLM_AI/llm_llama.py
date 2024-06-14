import json
import os.path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

from LLM_AI.llm_base import ForbiddenRomanNumbersLogitsProcessor, SuppressSpecificBOSTokenLogitsProcessor, ForceTokenFixValueLogitsProcessor, \
    SentenceStoppingCriteria
from et_base import check_multi_head_attention
from dataclasses import dataclass
from typing import Dict
import warnings
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from et_dirs import llm_ai_base, model_dir_base
models_dir = os.path.join(os.path.join(model_dir_base, os.path.basename(llm_ai_base)), 'models')

from et_base import ET_LLM

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class Template:
    template_name: str
    system_format: str
    system: str
    user_format: str
    user_complete: str
    assistant_format: str
    assistant_complete: str
    stop_word: str


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, system, user_format, user_complete,
                      assistant_format, assistant_complete, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        system=system,
        user_format=user_format,
        user_complete=user_complete,
        assistant_format=assistant_format,
        assistant_complete=assistant_complete,
        stop_word=stop_word,
    )


register_template(
    template_name='llama3',
    system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    system="You are a super seller. You are selling products in air now. \n\nYou can only rely in English.",
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    user_complete='<|start_header_id|>user<|end_header_id|>\n\n',
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant_complete='<|start_header_id|>assistant<|end_header_id|>\n\n',
    stop_word='<|eot_id|>'
)


def load_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device,
        quantization_config=None
    )
    return model


def load_tokenizer(model_name_or_path, add_prefix_space=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        add_prefix_space=add_prefix_space
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(tokenizer, template, query, history, system=None):
    # 模板内容
    # template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    # user_complete = template.user_complete
    assistant_format = template.assistant_format
    assistant_complete = template.assistant_complete
    system = system if system else template.system
    # 添加系统信息
    system_text = system_format.format(content=system)
    input_ids = tokenizer.encode(system_text, add_special_tokens=True)
    # 拼接历史对话
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=True)
        input_ids += tokens
    # 用户输入问题，要求llm输出回复内容
    # history.append({"role": 'user', 'message': query})
    qa_text = user_format.format(content=query, stop_token=tokenizer.eos_token)
    tokens = tokenizer.encode(qa_text, add_special_tokens=True)
    qa_ids_tokens = len(tokens)
    input_ids += tokens
    # history.append({"role": 'assistant', 'message': ''})
    complete_text = assistant_complete.format(content='', stop_token=tokenizer.eos_token)
    tokens = tokenizer.encode(complete_text, add_special_tokens=True)
    input_ids += tokens
    # 返回encode的ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    return input_ids, qa_ids_tokens


class LLM_Llama_V3(ET_LLM):
    def __init__(self, model_name: str = 'LLM-Research/Meta-Llama-3-8B-Instruct', temp_name: str = 'llama3'):
        super().__init__()
        self.template_name = temp_name
        assert self.template_name in template_dict
        self.template = template_dict[self.template_name]
        model_path = os.path.join(models_dir, model_name)
        assert os.path.exists(model_path)
        self.model_path = model_path
        # 采用lazy_load加载
        self.model = None
        self.tokenizer = None
        self.stop_token_id = None
        self.infer_dict = None
        self.sentence_token_list = None
        self.history_cached = None
        # # bad_words_ids
        self.roman_numbers_list = None
        self.forbidden_roman_numbers_logits_processor = None
        self.suppress_specific_BOS_token_logits_processor = None
        # 中英文控制
        self._masked_indicator_cn = None
        self._masked_indicator_en = None

    def lazy_load(self):
        self.model = load_model(self.model_path).eval()
        self.tokenizer = load_tokenizer(self.model_path)
        if self.template.stop_word is None:
            self.template.stop_word = self.tokenizer.eos_token
        stop_token_id = self.tokenizer.encode(self.template.stop_word, add_special_tokens=True)
        assert len(stop_token_id) >= 1
        self.stop_token_id = stop_token_id[0]
        # 默认推理参数
        self.infer_dict = {
            'max_new_tokens': 256,
            'min_new_tokens': 8,
            'top_p': 0.75,
            'top_k': 3,
            'temperature': 0.7,
            'repetition_penalty': 1.2,
            'eos_token_id': self.stop_token_id,
            'pad_token_id': self.tokenizer.eos_token_id,
            'do_sample': True,
            'no_repeat_ngram_size': 8,
            'length_penalty': 1.5,
            # 'max_length': 4096
        }
        # 0 means "!", # 13 means ".",# 30 means "?"
        self.sentence_token_list = ['!', '！', '.', '。', '?', '？',]
        # print(self.sentence_token_id_list)
        self.history_cached: dict[str, list] = {}
        # 罗马数字: [40, 5660, 23440, 3166, 53, 26376, 5660, 23440, 5511, 55, 84214, 5660, 23440, 3166, 53, 3166, 5660, 23440]
        self.roman_numbers_list = [
            " I", " II", " III", " IV", " V", " VI", " VII", " VIII", " IX", " X", " XI", " XII", " XIII", " XIV", " XV", " XVI", " XVII"
            " XXIV", " XXXV", " XXVII", " XXXIII",
        ]
        self.forbidden_roman_numbers_logits_processor = ForbiddenRomanNumbersLogitsProcessor(self.roman_numbers_list, self.tokenizer)
        punctuation_list = ['，', '。', '？', '！', '“', '”', '：', ',', '.', '?', '!', '"', "'", ':']
        punctuation_token_id_list = self.tokenizer.convert_tokens_to_ids(punctuation_list)
        self.suppress_specific_BOS_token_logits_processor = SuppressSpecificBOSTokenLogitsProcessor(punctuation_token_id_list)
        # 中文
        from et_base import is_chinese
        self._masked_indicator_cn = [token_id for token, token_id in self.tokenizer.vocab.items() if is_chinese(token)]
        # self._masked_indicator_cn.extend(punctuation_token_id_list)
        self._masked_indicator_cn.extend([self.stop_token_id, self.tokenizer.eos_token_id])
        # 英文
        from et_base import is_english
        self._masked_indicator_en = [token_id for token, token_id in self.tokenizer.vocab.items() if is_english(token)]
        # self._masked_indicator_en.extend(punctuation_token_id_list)
        self._masked_indicator_en.extend([self.stop_token_id, self.tokenizer.eos_token_id])

    def unload_model(self):
        if self.model:
            self.model = self.tokenizer = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def is_load(self):
        return self.model is not None

    def llm(self, query: str, system: str, **kwargs):
        check_multi_head_attention()
        if not self.model:
            self.lazy_load()
        # 参数判断
        # if not query or query.strip() == '':
        #     return ''
        print(f"query=>{system}\n\n{query}")
        query = query.strip()
        # 历史记录，通过uuid绑定
        uuid_key = kwargs['et_uuid'] if 'et_uuid' in kwargs else None
        history = self.history_cached[uuid_key] if uuid_key is not None and uuid_key in self.history_cached else []
        if len(history) > 0: history = history.copy()
        # 构建参数
        infer_params = self.infer_dict.copy()
        if kwargs:  # 只接受配置参数
            for k, v in infer_params.items():
                if k in kwargs: infer_params[k] = kwargs[k]
        self.tokenizer.encode(query, add_special_tokens=True)
        # 最大句子数限制
        max_num_sentence = 3 if 'max_num_sentence' not in kwargs else kwargs['max_num_sentence']
        max_num_sentence = min(max(max_num_sentence, 1), 6)
        # prompt编码
        input_ids, qa_ids_tokens = build_prompt(self.tokenizer, self.template, query=query,
                                                system=system, history=history)
        # 调整max_new_tokens/min_new_tokens参数
        max_new_tokens = 256 if 'max_new_tokens' not in kwargs else kwargs['max_new_tokens']
        max_new_tokens = max(max_new_tokens, qa_ids_tokens)
        min_new_tokens = max_new_tokens//2 if 'min_new_tokens' not in kwargs else kwargs['min_new_tokens']
        min_new_tokens = max(min_new_tokens, qa_ids_tokens//2)
        infer_params['max_new_tokens'] = max_new_tokens
        infer_params['min_new_tokens'] = min_new_tokens
        sentence_stopping_criteria = SentenceStoppingCriteria(max_num_sentence=max_num_sentence,
                                                              sentence_token_list=self.sentence_token_list,
                                                              dot_token=self.sentence_token_list[1],
                                                              stop_token_id_list=[self.stop_token_id],
                                                              stop_word=self.template.stop_word,
                                                              tokenizer=self.tokenizer)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(sentence_stopping_criteria)
        logits_processor = LogitsProcessorList()
        logits_processor.append(self.forbidden_roman_numbers_logits_processor)
        logits_processor.append(self.suppress_specific_BOS_token_logits_processor)
        language = kwargs['language'] if 'language' in kwargs else None
        if language == 'chinese':
            logits_processor.append(ForceTokenFixValueLogitsProcessor(self._masked_indicator_en))
        elif language == 'english':
            logits_processor.append(ForceTokenFixValueLogitsProcessor(self._masked_indicator_cn))
        print('language =', language)
        # 开始推理
        outputs = self.model.generate(input_ids=input_ids, logits_processor=logits_processor,
                                      stopping_criteria=stopping_criteria, **infer_params)
        outputs = outputs.tolist()
        # print(f'len(outputs)={len(outputs)}')
        outputs = outputs[0][len(input_ids[0]):]
        if sentence_stopping_criteria.last_sentence_token_idx > 0:
            outputs = outputs[:sentence_stopping_criteria.last_sentence_token_idx]
        print('reason_stop ==>', sentence_stopping_criteria.reason_stop)
        an = ''.join(sentence_stopping_criteria.tokens_decoded_words)
        if an == '': an = self.tokenizer.decode(outputs)
        idx = an.rfind(self.template.stop_word)
        if idx > 0: an = an[:idx]
        idx = max([an.rfind(_ch) for _ch in self.sentence_token_list])
        if idx > 0: an = an[:idx+1]
        an = an.replace('\n', '').strip().replace(self.template.stop_word, "").strip()
        an = an.replace('<|start_header_id|>', '').replace('assistant<|end_header_id|>', '')
        an = an.replace('>', '').strip()
        # 历史记录
        history.append({"role": 'user', 'message': query})
        history.append({"role": 'assistant', 'message': an})
        if uuid_key is not None: self.history_cached[uuid_key] = history
        # 返回结果
        print("infer<=", an)
        return an
