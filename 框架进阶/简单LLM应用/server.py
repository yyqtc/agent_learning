from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

system_prompt = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_template(
    ('system', system_prompt),
    ('user', '{text}')
)

model = ChatOpenAI(
    model = "deepseek-chat",
    openai_api_key=config["DEEPSEEK_API_KEY"],
    openai_api_base="https://api.deepseek.com",
    temperature=0.7,
    max_tokens=2000
)

parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(
    title="LangChain Server",
    version="1.0",
)
