import json

config = json.load(open("config.json", "r"))

from langchain_community.document_loaders import PyPDFLoader

file_path = "C:\\Users\\80570\\Desktop\\test.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=True
)

with open("store-usage-documentation.md", "r", encoding="utf-8") as f:
    markdown_content = f.read()
    all_splits = text_splitter.split_text(markdown_content)
    
    from langchain.vectorstores import Chroma
    from langchain.embeddings import DashScopeEmbeddings
    from uuid import uuid4
    
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=config["QWEN_API_KEY"]
    )
    vectorstore = Chroma(
        collection_name="test",
        embedding_function=embeddings
    )
    uuids = [str(uuid4()) for _ in range(len(all_splits))]
    vectorstore.add_documents(documents=all_splits, ids=uuids)
    result = vectorstore.similarity_search_with_score("什么是LangChain？")
    
    print(result)
