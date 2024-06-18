import re

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

class ForbiddenRomanNumbersLogitsProcessor(LogitsProcessor):
    def __init__(self, roman_token_id_list: list[str], tokenizer):
        super().__init__()
        self._roman_token_id_list = roman_token_id_list
        self._model_tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_word = self._model_tokenizer.decode(input_ids[0][-1]).lower()
        if 'type' in last_word:
            processed_scores = scores.clone()
            processed_scores[:, self._roman_token_id_list] = -float('inf')
            return processed_scores
        else:
            return scores


class ForbiddenFollowingTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, rule_dict: dict[str, list[int]], tokenizer):
        self._rule_dict = rule_dict
        self._model_tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_word = self._model_tokenizer.decode(input_ids[0][-1]).lower()
        processed_scores = scores.clone()
        for key, value in self._rule_dict.items():
            if key in last_word: processed_scores[:, value] = -float('inf')
        return processed_scores


class EncourageFollowingTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, rule_dict: dict[str, list[int]], tokenizer, factor: float=2.0):
        self._rule_dict = rule_dict
        self._model_tokenizer = tokenizer
        self._factor = factor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_word = self._model_tokenizer.decode(input_ids[0][-1]).lower()
        processed_scores = scores.clone()
        for key, value in self._rule_dict.items():
            if key in last_word: processed_scores[:, value] *= self._factor
        return processed_scores


class ForbiddenLeadingPunctuationsLogitsProcessor(LogitsProcessor):
    def __init__(self, bad_bos_token_id_list, start_ids_length):
        self._bad_bos_token_id_list = [token_id for token_id in bad_bos_token_id_list if token_id is not None]
        self._start_ids_length = start_ids_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_token_len = input_ids.shape[-1]
        if new_token_len <= self._start_ids_length:
            processed_scores = scores.clone()
            processed_scores[:, self._bad_bos_token_id_list] = -float('inf')
            return processed_scores
        else:
            return scores

class ForceTokenFixValueLogitsProcessor(LogitsProcessor):
    def __init__(self, language_token_id_list, value_to_set=-float('inf')):
        super().__init__()
        self._language_token_id_list = language_token_id_list
        self._value_to_set = value_to_set

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        processed_scores = scores.clone()
        processed_scores[:, self._language_token_id_list] = self._value_to_set
        return processed_scores


class SentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, max_num_sentence: int, sentence_token_list: [int], dot_token: int,
                 stop_token_id_list: [int], stop_word: str, tokenizer):
        self._max_num_sentence = max_num_sentence
        self._sentence_token_list = sentence_token_list
        self._sentence_pattern = r'['+''.join(self._sentence_token_list)+']|\.'
        self._count_num_sentence = 0
        self._stop_token_id_list = stop_token_id_list
        self._stop_word = stop_word
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
        return self._model_tokenizer.decode(token_id, add_special_tokens=True)

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
            else:  # 其他情况
                self._count_num_sentence += 1
            # 判断句子数量
            if self._count_num_sentence >= self._max_num_sentence:
                self.reason_stop = f'max_sentence_1({self._count_num_sentence})'
                return True
        else:
            count_sentence = self._count_sentence(self.tokens_decoded_words[-1])
            if count_sentence > 0:
                self._count_num_sentence += count_sentence
                # 判断句子数量
                if self._count_num_sentence >= self._max_num_sentence:
                    self.reason_stop = f'max_sentence_2({self._count_num_sentence})'
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
