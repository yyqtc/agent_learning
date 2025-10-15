from langchain_chroma import Chroma
from langchain.embeddings import DashScopeEmbeddings
from langchain.docstore.document import Document
from uuid import uuid4

import json

config = json.load(open("config.json", encoding="utf-8"))

_embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=config["QWEN_API_KEY"]
)

_vs = Chroma(
    collection_name="query_history",
    embedding_function=_embeddings
)

def add_history(query: str, answer: str):
    doc = Document(
        page_content=query,
        metadata={"source": "agent answer", "answer": answer},
        id=str(uuid4())
    )
    _vs.add_documents([doc])

def search_history(query: str):
    return _vs.similarity_search_with_score(query, k=5)
