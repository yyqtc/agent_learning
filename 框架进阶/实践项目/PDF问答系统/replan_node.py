from langgraph.graph import END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from custome_type import Plan, Response, Act, PlanExecute
from typing import Union

import json

config = json.load(open("config.json", encoding="utf-8"))

_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一个擅于根据当前计划执行结果分析计划是否合理的助手。
        
        我们的目标曾是：
        {input}

        我们的最近一次计划是：
        {plan}

        我们已经完成了以下步骤：
        {past_steps}

        根据以上信息更新我们的计划。如果你认为不需要执行更多步骤，你可以直接输出答案给用户。否则你需要在计划中补充更多步骤。
        你应该判断我们的目标是否和本系统存储的PDF文件有关，如果无关，直接输出“本系统无法回答你的这个问题”给用户。
        你输出的计划内的步骤应该是相互独立的。
        你输出的计划不能包含与我们的目标无关的步骤。
        你输出的计划不应该有任何和答案有关的暗示。
        你输出的计划中不应该包含已经完成的步骤。
        你输出的计划中每个步骤都必须确保这个步骤能够得到所有它需要的信息。
        你输出的计划应该确保最后一个步骤输出的是对用户问题的最终答案。
        计划请以JSON格式输出，包含action字段，action字段应该包含steps字段，steps字段类型List[str]。
        答案请以JSON格式输出，包含action字段，action字段应该包含response字段，response字段类型str。
        """
    )
])

_llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=config["QWEN_API_KEY"],
    openai_api_base=config["QWEN_API_BASE"],
    temperature=0
)

replanner = _prompt | _llm.with_structured_output(Act)

async def replan_node(state: PlanExecute) -> Union[PlanExecute, END]:
    result = await replanner.ainvoke(state)

    if isinstance(result.action, Response):
        return {
            "response": result.action.response
        }
    else:
        return {
            "plan": result.action.steps
        }
