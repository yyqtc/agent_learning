from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import DashScopeEmbeddings
from langchain.docstore.document import Document
from uuid import uuid4
from typing import List, Dict

import json
import os

config = json.load(open("config.json", "r", encoding="utf-8"))

def _init_vector_store(path: str, collection_name: str) -> Chroma:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=config["QWEN_API_KEY"]
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )
    all_splits = []
    try:
        if not os.path.exists(path):
            return None

        if os.path.isdir(path):
            files = _traverse_directory(path)
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    splits = text_splitter.split_text(content)
                    all_splits = [*all_splits, *splits]
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                splits = text_splitter.split_text(content)
                all_splits = [*all_splits, *splits]
        
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        vectorstore.add_documents(documents=all_splits, ids=uuids)
        return vectorstore
    except Exception as e:
        print(f"初始化向量存储时出错: {e}")
        return None


def _init_vector_store_without_split(path: str, collection_name: str) -> Chroma:
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=config["QWEN_API_KEY"]
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )
    all_docs = []
    try:
        if not os.path.exists(path):
            return None

        if os.path.isdir(path):
            files = _traverse_directory(path)
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = Document(page_content=content, metadata={"source": file})
                    all_docs.append(doc)
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"source": file})
                all_docs.append(doc)
        
        uuids = [str(uuid4()) for _ in range(len(all_docs))]
        vectorstore.add_documents(documents=all_docs, ids=uuids)
        return vectorstore
    
    except Exception as e:
        print(f"初始化向量存储时出错: {e}")
        return None


def _traverse_directory(path: str) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                files.append(file_path)
            
            if os.path.isdir(file_path):
                temp_files = _traverse_directory(file_path)
                files = [*files, *temp_files]

    return files


def init_all_db() -> List[Dict]:
    vector_stores_wrapper: List[Dict] = []
    requirement_db = _init_vector_store(config["REQUIREMENT_DB_PATH"], "requirement")
    vector_stores_wrapper.append({
        "db": requirement_db,
        "description": "本次前端开发需求文档的向量数据库"
    })
    api_db = _init_vector_store(config["API_DB_PATH"], "api")
    vector_stores_wrapper.append({
        "db": api_db,
        "description": "已有vue+uniapp前端项目的/src/api目录下的api调用情况的说明文档"
    })
    common_utils_db = _init_vector_store(config["COMMON_UTILS_DB_PATH"], "common_utils")
    vector_stores_wrapper.append({
        "db": common_utils_db,
        "description": "已有vue+uniapp前端项目的/src/common/utils目录下的公共工具方法的说明文档"
    })
    storage_db = _init_vector_store(config["STORAGE_DB_PATH"], "storage")
    vector_stores_wrapper.append({
        "db": storage_db,
        "description": "已有vue+uniapp前端项目的页面中使用localStorage、sessionStorage、uni.xxxStorageSync的说明文档"
    })
    store_db = _init_vector_store(config["STORE_DB_PATH"], "store")
    vector_stores_wrapper.append({
        "db": store_db,
        "description": "已有vue+uniapp前端项目的/src/store/index.js中全局状态变量及其修改方法的说明文档"
    })
    utils_db = _init_vector_store(config["UTILS_DB_PATH"], "utils")
    vector_stores_wrapper.append({
        "db": utils_db,
        "description": "已有vue+uniapp前端项目的/src/utils目录下的工具方法的说明文档"
    })

    return vector_stores_wrapper

if __name__ == "__main__":
    init_all_db()