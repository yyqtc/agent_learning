from langgraph.graph import StateGraph, START, END
from custome_type import PlanExecute
from plan_node import plan_node
from execute_node import execute_node
from replan_node import replan_node
from db import add_history

import asyncio

def _should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        add_history(state["input"], state["response"])
        return END
    else:
        return "execute"

async def main():
    workflow = StateGraph(PlanExecute)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("replan", replan_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "replan")
    workflow.add_conditional_edges(
        "replan",
        _should_end,
        ["execute", END]
    )

    app = workflow.compile()

    while True:
        user_input = input("请输入问题: ")
        if user_input == "exit":
            break
        
        result = await app.ainvoke({
            "input": user_input
        }, config={"recursion_limit": 100})
        
        print(result["response"])

if __name__ == "__main__":
    asyncio.run(main())