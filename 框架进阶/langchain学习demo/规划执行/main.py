from langchain_community.tools.tavily_search import TavilySearchResults
import json
import os

config = json.load(open("config.json"))

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = config[var]

_set_env("TAVILY_API_KEY")

from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=config["QWEN_API_KEY"],
    openai_api_base=config["QWEN_API_BASE"],
    temperature=0.7,
    max_tokens=2000
)

prompt = "You are a helpful assistant."
agent_executor = create_react_agent(llm, tools, prompt=prompt)

import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

from pydantic import BaseModel, Field

class Plan(BaseModel):
    steps: List[str] = Field(
        description="将要执行的步骤，确保步骤按照先后顺序排序"
    )

from langchain_core.prompts import ChatPromptTemplate

planner = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        针对给定的目标，输出一份简单的按步骤执行的计划。
        这个计划应该包含若干个独立的任务，任务正确执行后，应该能够实现目标。
        不允许输出任何无关的任务。
        最后一个步骤输出的结果应该是最终的答案，确保每一个步骤都能得到所有需要的信息
        确保每个步骤都会执行。
        请以JSON格式输出计划，包含steps字段，steps字段类型List[str]。
        """
    ),
    ("placeholder", "{messages}")
]) | ChatOpenAI(
    model="qwen-plus", 
    openai_api_key=config["QWEN_API_KEY"], 
    openai_api_base=config["QWEN_API_BASE"],
    temperature=0
).with_structured_output(Plan)

from typing import Union

class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="将要执行的动作，可以是Response也可以是Plan。如果你想输出最终答案，使用Response。如果你认为在得到最终答案之前还需要调用一些工具，使用Plan"
    )

replanner = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        针对给定的目标，输出一份简单的按步骤执行的计划。
        这个计划应该包含若干个独立的任务，任务正确执行后，应该能够实现目标。
        不允许输出任何无关的任务。
        最后一个步骤输出的结果应该是最终的答案，确保每一个步骤都能得到所有需要的信息
        确保每个步骤都会执行。
        
        你的目标曾是：
        {input}

        你的最近一次计划是：
        {plan}

        你已经完成了以下步骤：
        {past_steps}

        根据以上信息更新你的计划。如果你认为不需要执行更多步骤，你可以直接输出答案给用户。否则你需要在计划中补充更多步骤。
        你只应该在计划中补充新的需要被执行的步骤，已经被完成的步骤不要补充进计划。
        计划请以JSON格式输出，包含action字段，action字段应该包含steps字段，steps字段类型List[str]。
        答案请以JSON格式输出，包含action字段，action字段应该包含response字段，response字段类型str。
        """
    )
]) | ChatOpenAI(
    model="qwen-plus",
    openai_api_key=config["QWEN_API_KEY"],
    openai_api_base=config["QWEN_API_BASE"],
    temperature=0
).with_structured_output(Act)

from typing import Literal
from langgraph.graph import END

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if not len(plan):
        return END
    
    plan_str = '\n'.join(f'{i + 1}. {step}' for i, step in enumerate(plan))
    task = plan.pop(0)
    print("task：", task, "\n")
    task_formatted = \
        f"""
        完成这个任务：{task}。不要返回和{task}无关的内容。
        """

    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    return {
        "past_steps": [(task, agent_response["messages"][-1].content)]
    }

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({
        "messages": [
            ("user", state["input"])
        ]
    })

    print("plan：\n", plan.steps, "\n")

    return {
        "plan": plan.steps
    }

async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {
            "response": output.action.response
        }
    else:
        return {
            "plan": output.action.steps
        }

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)
workflow.add_node("plan", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "plan")
workflow.add_edge("plan", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END]
)

app = workflow.compile()

async def test(str):
    inputs = {
        "input": str
    }
    result = await app.ainvoke(inputs, config={"recursion_limit": 100})
    if result["response"].strip():
        return result["response"]
    else:
        return "没有找到相关信息"

import asyncio

if __name__ == "__main__":
    result = asyncio.run(test("""
        帮我写一首七言绝句歌颂古四川人开拓精神，
        风格磅礴大气，
        语言含蓄古典，
        内容逻辑连贯，
        适当用典，
        意象奇而不诡，险而有据，
        谙地理，避陈言，
        避免使用现代词汇和网络词汇，
        第三句使用巴江倒卷云峰裂
    """))
    
    #result = asyncio.run(test("""
    #    我想开发一个开发Vue+Uniapp前端项目的agent，这个agent采用plan-and-execute框架，
    #    要确保planner具备上下文理解能力，能根据项目结构（如pages.json、components目录）和Vue+Uniapp框架规范生成合理任务序列的能力，我应该怎么开发planner，能否给一些提示
    #"""))

    print("result: \n", result, "\n")
