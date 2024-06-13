import os.path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


class SentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, max_num_sentence: int, sentence_token_list: [int], dot_token: int,
                 stop_token_id_list: [int], stop_word: str, tokenizer):
        self._max_num_sentence = max_num_sentence
        self._sentence_token_list = sentence_token_list
        self._sentence_pattern = r'[' + ''.join(self._sentence_token_list) + ']|\.'
        self._count_num_sentence = 0
        self._stop_token_id_list = stop_token_id_list
        self._stop_word = stop_word
        if not self._stop_token_id_list: self._stop_token_id_list = []
        self.last_sentence_token_idx = 0
        self._count_token = 0
        self.tokens_decoded_words = []
        # 过滤bad cases
        self._model_tokenizer = tokenizer
        self._sentence_token_id_list = [self._model_tokenizer.encode(_ch, add_special_tokens=False)[-1] for _ch in self._sentence_token_list]
        if not self._sentence_token_id_list: self._sentence_token_id_list = []
        self._dot_token_id = self._model_tokenizer.encode(dot_token, add_special_tokens=False)[-1]
        # 停止理由
        self.reason_stop = 'max_new_token'

    def _count_sentence(self, text):
        matches = re.findall(self._sentence_pattern, text)
        return len(matches)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        current_token_id = input_ids[0][-1].detach().cpu().numpy()
        current_token_id = current_token_id[()]
        try:
            self.tokens_decoded_words.append(self._model_tokenizer.decode(current_token_id))
        except TypeError:
            value = self._model_tokenizer._convert_id_to_token(current_token_id).decode("utf-8", errors="replace")
            self.tokens_decoded_words.append(value)
        # 开始判断逻辑
        if current_token_id in self._sentence_token_id_list:  # 当前是!.?其中一个
            if current_token_id == self._dot_token_id:  # 处理小数点
                if not str.isdigit(self.tokens_decoded_words[-2]):
                    self._count_num_sentence += 1
                    self.last_sentence_token_idx = self._count_token + 1
            else:  # 其他情况
                self._count_num_sentence += 1
                self.last_sentence_token_idx = self._count_token + 1
            # 判断句子数量
            if self._count_num_sentence >= self._max_num_sentence:
                self.reason_stop = 'max_sentence'
                return True
        else:
            count_sentence = self._count_sentence(self.tokens_decoded_words[-1])
            if count_sentence > 0:
                self._count_num_sentence += count_sentence
                self.last_sentence_token_idx = self._count_token + 1
                # 判断句子数量
                if self._count_num_sentence >= self._max_num_sentence:
                    self.reason_stop = 'max_sentence'
                    return True
        # 判断结束符号
        if current_token_id in self._stop_token_id_list:
            self.reason_stop = 'eos_token'
            return True
        if self._stop_word in self.tokens_decoded_words[-1]:
            self.reason_stop = 'stop_word'
            return True
        # 计算token数量
        self._count_token += 1
        return False


class LLM_GLM_4(ET_LLM):
    def __init__(self, model_name: str = 'ZhipuAI/glm-4-9b-chat', temp_name: str = 'glm4'):
        super().__init__()
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

    def lazy_load(self):
        self.model = load_model(self.model_path).eval()
        self.tokenizer = load_tokenizer(self.model_path)
        self.stop_token_id = [151329, 151336, 151338]
        # 默认推理参数
        self.infer_dict = {
            'max_new_tokens': 256,
            'min_new_tokens': 8,
            'top_p': 0.75,
            'top_k': 3,
            'temperature': 0.95,
            'repetition_penalty': 1.5,
            'eos_token_id': self.stop_token_id[0],
            'pad_token_id': self.tokenizer.eos_token_id,
            'do_sample': True,
            'no_repeat_ngram_size': 8,
            'length_penalty': 1.5,
            # 'max_length': 4096
        }
        self.sentence_token_list = ['!', '！', '.', '。', '?', '？', ':', '：']
        # print(self.sentence_token_id_list)
        self.history_cached = []

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
        # 最大句子数限制
        max_num_sentence = 3 if 'max_num_sentence' not in kwargs else kwargs['max_num_sentence']
        max_num_sentence = min(max(max_num_sentence, 1), 6)
        # prompt编码
        history.insert(0, {"role": "system", "content": system})
        history.append({"role": "user", "content": query})
        input_ids = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=True,
                                                       return_tensors="pt", return_dict=True)
        input_ids = input_ids.to(device)
        # 调整max_new_tokens/min_new_tokens参数
        qa_ids = self.tokenizer.apply_chat_template([], add_generation_prompt=True, tokenize=True,
                                                    return_tensors="pt", return_dict=True)
        qa_ids_tokens = qa_ids['input_ids'].shape[1]
        max_new_tokens = 256 if 'max_new_tokens' not in kwargs else kwargs['max_new_tokens']
        max_new_tokens = max(max_new_tokens, qa_ids_tokens)
        min_new_tokens = max_new_tokens // 2 if 'min_new_tokens' not in kwargs else kwargs['min_new_tokens']
        min_new_tokens = max(min_new_tokens, qa_ids_tokens // 2)
        infer_params['max_new_tokens'] = max_new_tokens
        infer_params['min_new_tokens'] = min_new_tokens
        sentence_stopping_criteria = SentenceStoppingCriteria(max_num_sentence=max_num_sentence,
                                                              sentence_token_list=self.sentence_token_list,
                                                              dot_token=self.sentence_token_list[1],
                                                              stop_token_id_list=self.stop_token_id,
                                                              stop_word='<|endoftext|>',
                                                              tokenizer=self.tokenizer)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(sentence_stopping_criteria)
        outputs = self.model.generate(**input_ids, stopping_criteria=stopping_criteria, **infer_params)
        outputs = outputs[:, input_ids['input_ids'].shape[1]:]
        if sentence_stopping_criteria.last_sentence_token_idx > 0:
            outputs = outputs[:sentence_stopping_criteria.last_sentence_token_idx]
        print('reason_stop ==>', sentence_stopping_criteria.reason_stop)
        an = ''.join(sentence_stopping_criteria.tokens_decoded_words)
        if an == '': an = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        an = an.strip().strip('\n')
        # 当不存在<|endoftext|> <|user|> <|observation|>时，就是主动停止推理，否则就是达到max_tokens停止推理
        idx = an.rfind('<|endoftext|>')
        if idx > 0: an = an[:idx]
        # 历史记录
        history[-1] = {"role": 'user', 'message': query}
        history.append({"role": 'assistant', 'message': an})
        if uuid_key is not None: self.history_cached[uuid_key] = history[1:]
        # 返回结果
        print("infer<=", an)
        return an
