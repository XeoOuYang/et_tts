import re

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

class ForbiddenRomanNumbersLogitsProcessor(LogitsProcessor):
    def __init__(self, bad_words: list[str], tokenizer):
        super().__init__()
        self._model_tokenizer = tokenizer
        self._bad_word_token_id_list = [self._model_tokenizer.encode(_word, add_special_tokens=False)[-1] for _word in bad_words]
        self._indicator_to_set = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_word = self._model_tokenizer.decode(input_ids[0][-1])
        if 'type' in last_word.lower():
            scores[:, self._bad_word_token_id_list] = -float('inf')
        return scores


class SuppressSpecificBOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, bad_bos_token_id_list = None):
        self.bad_bos_token_id_list = bad_bos_token_id_list
        self._indicator_to_set = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_token_len = input_ids.shape[-1]
        if new_token_len == 0:
            scores[:, self.bad_bos_token_id_list] = -float('inf')
        return scores

class ForceTokenFixValueLogitsProcessor(LogitsProcessor):
    def __init__(self, language_token_id_list, value_to_set: float=-float('inf')):
        super().__init__()
        self._language_token_id_list = language_token_id_list
        self._value_to_set = value_to_set
        self._indicator_to_set = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores[:, self._language_token_id_list] = self._value_to_set
        return scores


class SentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, max_num_sentence: int, sentence_token_list: [int], dot_token: int,
                 stop_token_id_list: [int], stop_word: str, tokenizer):
        self._max_num_sentence = max_num_sentence
        self._sentence_token_list = sentence_token_list
        self._sentence_pattern = r'['+''.join(self._sentence_token_list)+']|\.'
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

    def _decode_token(self, token_id):
        return self._model_tokenizer.decode(token_id)

    def _current_token_id(self, input_ids):
        current_token_id = input_ids[0][-1].detach().cpu().numpy()
        return current_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        current_token_id = self._current_token_id(input_ids)
        self.tokens_decoded_words.append(self._decode_token(current_token_id))
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
