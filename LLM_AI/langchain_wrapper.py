import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessor, LogitsProcessorList
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

from et_dirs import llm_ai_base, model_dir_base
models_dir = os.path.join(os.path.join(model_dir_base, os.path.basename(llm_ai_base)), 'models')

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="<|start_header_id|>system<|end_header_id|>\n\n"
             "你是一名主播，请根据以下提供的内容，回复用户。\n{context}\n必须以第一人称，用中文回复用户。<|eot_id|>"
             "<|start_header_id|>user<|end_header_id|>\n\n"
             "用户：{question}<|eot_id|>"
             "<|start_header_id|>assistant<|end_header_id|>\n\n"
             "主播："
)

model_name = 'LLM-Research/Meta-Llama-3-8B-Instruct'
model_path = os.path.join(models_dir, model_name)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False,
    add_prefix_space=False
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    quantization_config=None
)

class ConditionForcedEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, stop_token_id):
        self._tokenizer = tokenizer
        self._stop_token_id = stop_token_id
        self._decode_token: str = ""
        self._start_token = -1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self._start_token < 0:
            self._start_token = input_ids.shape[-1]
        else:
            self._decode_token = self._tokenizer.decode(input_ids[0][self._start_token:])
            # 出现\n\n立刻停止
            # should_end = '\n\n' in self._decode_token
            # if should_end:
                # print(self._decode_token)
                # self._start_token = -1
                # scores_processed = torch.full_like(scores, -math.inf)
                # scores_processed[:, self._stop_token_id] = 0
                # return scores_processed
        # 否则继续
        return scores

stop_token_id = tokenizer.encode("<|eot_id|>", add_special_tokens=True)
debug = ConditionForcedEOSLogitsProcessor(tokenizer, stop_token_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    top_p=0.9,
    top_k=64,
    temperature=1.13,
    repetition_penalty=1.2,
    eos_token_id=stop_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    no_repeat_ngram_size=8,
    num_beams=2,
    length_penalty=1.5,
    logits_processor=LogitsProcessorList([debug])
)

# question = "保质期大概多久啊？"
#
# from chroma_wrapper import chromadb
# # 查询相似度
# from et_base import timer
# with timer('retrieve'):
#     context = ''
#     results = chromadb.similarity_search(question)
#     for idx, emb in enumerate(results):
#         context += f'{emb.page_content}\n\n'
#     context = context.strip('\n')
#
# prompt_params = {
#     "context": context,
#     "question": question,
# }

chain = HuggingFacePipeline(pipeline=pipe) | StrOutputParser()

def run_query(query):
    if query == 'exit':
        return None
    # 查询相似度
    from chroma_wrapper import chromadb, reranker
    # token相似度查询向量数据库
    results = chromadb.similarity_search(query)
    # 语义相关性重排序结果
    scores = reranker.compute_score([[query, emb.page_content] for emb in results], normalize=True)
    results = sorted([(r, s) for r, s in zip(results, scores)], key=lambda t: t[1], reverse=True)
    results = [result[0] for result in results if result[1] >= 0.01]
    context = ''
    for idx, emb in enumerate(results):
        context += f'{emb.page_content}\n\n'
    context = context.strip('\n')
    # 拼装prompt
    prompt_params = {
        "context": context,
        "question": query,
    }
    # langchain调用
    llm_input = prompt.invoke(prompt_params)
    results = chain.invoke(llm_input)
    # print(results)
    # print(debug._decode_token)
    return results[len(llm_input.text):]


# todo: history -> prompt
# todo: task prompt design
# todo: output parser
# todo: prd log
if __name__ == '__main__':
    while True:
        resp = run_query(input('用户：'))
        if resp == None:
            break
        else:
            from zhconv import zhconv
            resp = resp.strip()
            resp = zhconv.convert(resp, "zh-cn")
            resp = resp.replace(' ', '')
            resp = resp.replace('**', '')
            # target_index = resp.find('\n\n')
            # if target_index > 0: resp = resp[:target_index]
            resp = resp.replace('\n\n', '\n')
            print(f'主播：{resp}')