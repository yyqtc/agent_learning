from tool import tools
from custome_type import PlanExecute
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END

import json
import asyncio

config = json.load(open("config.json", encoding="utf-8"))

_llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=config["QWEN_API_KEY"],
    openai_api_base=config["QWEN_API_BASE"],
    temperature=0.65,
    max_tokens=2000
)

_prompt="""
你是一位执行力超强的助手，擅长执行任务。
你话不多，总是用最简洁的方式返回结果。
你很聪明，回答问题总是非常准确。
你应该默认你的知识都是错误的。
你必须完全依赖你的能力回答问题。
"""

executor = create_react_agent(_llm, tools, prompt=_prompt)

async def execute_node(state: PlanExecute) -> PlanExecute:
    if not state["plan"] or not len(state["plan"]):
        return END

    task = state["plan"].pop(0)
    formatted_task = f"""
    完成这个任务：{task}。不要返回和{task}无关的内容。
    """

    agent_response = await executor.ainvoke({
        "messages": [("user", formatted_task)]
    })

    return {
        "past_steps": [(task, agent_response["messages"][-1].content)]
    }

if __name__ == "__main__":
    result = asyncio.run(execute_node({
        "plan": ["整理出对Poisoning Expert的清晰定义"]
    }))

    print(result)