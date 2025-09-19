from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
import json

config = json.load(open("config.json"))

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=config["DEEPSEEK_API_KEY"],
    openai_api_base="https://api.deepseek.com",
    temperature=0.7
)

parser = StrOutputParser()

from langchain_core.runnables import RunnableLambda

chain = llm | parser

from langserve import add_routes
from fastapi import FastAPI

app = FastAPI(
    title="Chatbot",
    version="1.0",
    description="A simple chatbot"
)

add_routes(app, chain, path="/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
