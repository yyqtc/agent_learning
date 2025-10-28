from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from custome_type import Plan, PlanExecute

import json
import asyncio

config = json.load(open("config.json", encoding="utf-8"))

_prompt=ChatPromptTemplate.from_messages([
    (
        "system", 
        f"""
        你是一位擅长拆解任务的助手，擅长将一个复杂任务拆解为若干个简单的独立子任务。
        你应该默认认为用户的问题和本系统存储的PDF文件有关。
        你需要根据用户的问题，输出一个计划，这个计划中应该包含若干个独立的子任务，子任务正确执行后，应该能够回答用户的问题。
        你输出的计划应该首先判断是否和本系统存储的PDF文件有关，并告诉执行者用户的问题。
        你输出的计划中应该让执行者在做具体的pdf文件查询之前，先搜索查询记录向量数据库，加快重复查询的回答速度。
        计划列表中不允许出现和用户问题无关的子任务。
        你输出的计划不应该出现任何和答案有关的暗示。
        你如果没有从向量数据库中找到符合你要求的查询结果，请不要重复查询。
        确保计划中的每一步都能够得到所有需要的信息。
        确保计划的最后一步输出的是对用户问题的最终答案。
        请以JSON格式输出计划，包含steps字段，steps字段类型List[str]。
        """
    ),
    ("placeholder", "{messages}")
])

_llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=config["QWEN_API_KEY"],
    openai_api_base=config["QWEN_API_BASE"],
    temperature=0,
    max_tokens=2000
)

planner = _prompt | _llm.with_structured_output(Plan)

async def plan_node(state: PlanExecute) -> PlanExecute:
    result = await planner.ainvoke({
        "messages": [("user", state["input"])]
    })
    
    return {
        "plan": result.steps
    }

if __name__ == "__main__":
    result = asyncio.run(plan_node("什么是风控？"))
    print(result)
