from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tool import tools

import json
import time
import asyncio

config = json.load(open("./config.json", encoding="utf-8"))

def main():
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=config["DEEPSEEK_API_KEY"],
        openai_api_base=config["DEEPSEEK_API_BASE"],
        temperature=0.7
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=100,
        verbose=True,
        handle_parsing_errors=True
    )

    t = time.localtime()

    user_input = f"""
        你是一个非常专业的日报生成助手，你需要按步骤完成以下两个任务。
        1.你需要分析关键词为{",".join(config["KEYWORDS"])}的论文，生成论文日报。
        日报的标题为：{t.tm_year}年{t.tm_mon}月{t.tm_mday}日的论文日报。
        你应该默认你的知识是错误的，你必须完全依赖能力来完成任务。
        你生成日报的格式必须为：
        ```
        发送日期：{t.tm_year}年{t.tm_mon}月{t.tm_mday}日
        关键词：{",".join(config["KEYWORDS"])}
        今日新发布论文列表：
        论文的标题
            论文的作者
            论文的简单介绍
        ```
        2.你必须将日报以邮件形式发送到系统指定的邮箱。
    """

    asyncio.run(agent.ainvoke({
        "input": user_input
    }))

if __name__ == "__main__":
    main()
