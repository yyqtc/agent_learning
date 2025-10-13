from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import DashScopeEmbeddings
from langchain_core.tools import Tool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
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
        _llm = ChatOpenAI(
            model = "qwen-plus",
            openai_api_key=config["QWEN_API_KEY"],
            openai_api_base=config["QWEN_API_BASE"],
            temperature=0.7
        )

        vector_store_tools = self._init_vector_store_tool(vector_stores_wrapper)
        tools = [*vector_store_tools, *tools]

        self._memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self._agent = initialize_agent(
            tools=tools,
            llm =_llm,
            memory=self._memory,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=500,
            verbose=True,
            handle_parsing_errors=True
        )

    def _init_vector_store_tool(self, vector_stores_wrapper: List[Dict]) -> List[Callable]:
        tools = []
        for vs_wrapper in vector_stores_wrapper:
            description = vs_wrapper.get("description", "")
            tool_name = vs_wrapper.get("tool_name", "")
            db = vs_wrapper.get("db", None)
            if db is None or tool_name == "":
                continue

            retriever = db.as_retriever(search_kwargs={"k": 5})
            def make_search(r):
                def search(question: str):
                    docs = r.get_relevant_documents(question)
                    raw_texts = [doc.page_content for doc in docs]
                    result = "\n\n".join(raw_texts)
                    print("\n======== db search result begin ========\n")
                    print("question: " + question)
                    print("answer: \n" + result)
                    print("\n========  db search result end  ========\n")
                    return result
                return search

            tools.append(Tool(
                name=tool_name,
                func=make_search(retriever),
                description=description
            ))

        return tools
    
    def clear_memory(self):
        self._memory.clear()

    def run(self, user_input: str) -> str:
        """
        运行智能体，处理用户输入
        
        Args:
            user_input: 用户输入的问题或指令
            
        Returns:
            智能体的回复
        """
        try:
            logger.info(f"用户输入: {user_input}")
            
            # 调用 LangChain agent 处理用户输入
            response = self._agent.invoke({
                "input": user_input,
                "chat_history": self._memory.chat_memory.messages
            })
            final_output = response.get("output", response.get("return_value", str(response)))
            logger.info(f"智能体回复: {final_output}")

            return final_output
            
        except Exception as e:
            error_msg = f"处理用户输入时出错: {str(e)}"
            logger.error(error_msg)
            return error_msg
        