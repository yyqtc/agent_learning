### 尝试用Gemini + LangChain创作音乐

import os
import json

config = json.load(open("config.json"))

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=config["DEEPSEEK_API_KEY"],
    openai_api_base="https://api.deepseek.com",
    temperature=0.7,
    max_tokens=2000
)

prompt = ChatPromptTemplate.from_template(
    "你是一个英语翻译专家，请将以下内容翻译成中文：{input}"
)

chain = prompt | llm

response = chain.invoke({
    "role": "user",
    "style": "专业、简洁、准确",
    "input": "You are the silence I've been waiting for"
})

print(response.content)