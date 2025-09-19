import json

config = json.load(open("config.json", "r"))

from langchain_community.document_loaders import PyPDFLoader

file_path = "C:\\Users\\80570\\Desktop\\test.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=20, 
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

from langchain_community.embeddings import DashScopeEmbeddings

embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=config["QWEN_API_KEY"]
)

