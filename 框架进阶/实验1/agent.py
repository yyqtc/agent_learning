from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import DashScopeEmbeddings
from langchain_core.tools import Tool
from langchain.chains import RetrievalQA
from uuid import uuid4
from typing import List, Dict, Callable

import logging
import json

config = json.load(open("config.json", "r", encoding="utf-8"))

logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO 及以上会被记录
    format='%(asctime)s [%(levelname)s] %(message)s',  # 时间 | 等级 | 消息
    handlers=[
        logging.StreamHandler()  # 输出到终端
    ]
)
logger = logging.getLogger(__name__)  # 创建一个独立的 logger 实例

class Agent:
    def __init__(self, tools: List[Callable], vector_stores_wrapper: List[Dict]):
        self.todo_list = []

        _llm = ChatOpenAI(
            model = "deepseek-chat",
            openai_api_key=config["QWEN_API_KEY"],
            openai_api_base=config["QWen-API-BASE"],
            temperature=0.7
        )

        vector_store_tools = self._init_vector_store_tool(vector_stores_wrapper)
        tools = [*vector_store_tools, *tools]

        self._agent = initialize_agent(
            tools=tools,
            llm =_llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def _init_vector_store_tool(self, vector_stores_wrapper: List[Dict]) -> List[Callable]:
        tools = []
        for vs_wrapper in vector_stores_wrapper:
            description = vs_wrapper.get("description", "")
            db = vs_wrapper.get("db", None)
            if db is None:
                continue

            retriever = db.as_retriever(search_kwargs={"k": 5})
            _llm = ChatOpenAI(
                model="qwen-plus",
                openai_api_key=config["QWEN_API_KEY"],
                openai_api_base=config["QWen-API-BASE"],
                temperature=0
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=_llm, 
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True
            )
            tools.append(Tool(
                name="KnowledgeBaseQA",
                func=lambda question: qa_chain.invoke({"query": question})["result"],
                description=description
            ))

        return tools
    
    def 