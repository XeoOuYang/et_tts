import os.path

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from FlagEmbedding import FlagReranker

from et_dirs import llm_ai_base
documents_dir = os.path.join(llm_ai_base, 'documents')
txt_file = os.path.join(documents_dir, 'p_000.txt')
chromadb_dir = os.path.join(llm_ai_base, 'chromadb')

def load_text(file_path, encoding='utf8', autodetect_encoding=False):
    loader = TextLoader(file_path, encoding=encoding, autodetect_encoding=autodetect_encoding)
    return loader.load()

def split_documents(documents, chunk_size=256, chunk_overlap=16):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# shibing624/text2vec-base-multilingual: 多语种
# shibing624/text2vec-base-chinese-paraphrase： 中文
# shibing624/text2vec-base-chinese：中文
# nghuyong/ernie-2.0-base-en：英文
# nghuyong/ernie-3.0-base-zh：中文
# thenlper/gte-large：英文
# thenlper/gte-large-zh：中文
# BAAI/bge-base-zh-v1.5
# BAAI/bge-base-en-v1.5
# BAAI/bge-m3
from et_dirs import llm_ai_base, model_dir_base
models_dir = os.path.join(os.path.join(model_dir_base, os.path.basename(llm_ai_base)), 'embeddings')
model_name = "BAAI/bge-m3"
model_path = os.path.join(models_dir, model_name.replace('/', os.path.sep))
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    # model_name=model_name,
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# 初始化向量数据库
if os.path.exists(chromadb_dir):
    chromadb = Chroma(embedding_function=embeddings, persist_directory=chromadb_dir)
else:
    # 加载并分割文档
    chunks = split_documents(load_text(txt_file))
    chromadb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chromadb_dir)

# BAAI/bge-reranker-base
# BAAI/bge-reranker-v2-m3
reranker_name = "BAAI/bge-reranker-v2-m3"
reranker_path = os.path.join(models_dir, reranker_name.replace('/', os.path.sep))
reranker = FlagReranker(reranker_path, use_fp16=True)

if __name__ == '__main__':
    # 查询相似度
    from et_base import timer
    with timer('retrieve'):
        query = '哦哦。我想和咖啡一起喝，可以吗'
        results = chromadb.similarity_search(query)
        print('\n\n'.join([f'{idx}\n{result.page_content}' for idx, result in enumerate(results)]))
        scores = reranker.compute_score([[query, emb.page_content] for emb in results], normalize=True)
        print(scores)
        results = sorted([(r, s) for r, s in zip(results, scores)], key=lambda t: t[1], reverse=True)
        results = [result[0] for result in results if result[1] >= 0.01]
        print('\n\n'.join([f'{idx}\n{result.page_content}' for idx, result in enumerate(results)]))
