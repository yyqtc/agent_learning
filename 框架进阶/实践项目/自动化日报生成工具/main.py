from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tool import tools

import json
from datetime import date, timedelta

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


    t = date.today() - timedelta(days=1)

    user_input = f"""
        你是一个非常专业的日报生成助手，你需要按步骤完成以下两个任务。
        1.你需要分析关键词为{",".join(config["KEYWORDS"])}的论文，生成论文日报。
        如果你发现没有论文发布，你也应该生成标题为：{t.strftime("%Y年%m月%d日")}没有论文发布的日报。
        如果你发现有论文发布，则生成标题为：{t.strftime("%Y年%m月%d日")}的论文日报。
        你应该默认你的知识是错误的，你必须完全依赖能力来完成任务。
        你生成日报的格式必须为：
        ```
        发送日期：{t.strftime("%Y年%m月%d日")}
        关键词：{",".join(config["KEYWORDS"])}
        今日arxiv上新发布的论文：
        论文的标题
            论文的作者
            论文的简单介绍
        ```
        在邮件的末尾你应该提醒收取邮件的用户不要回复这封邮件，因为这封邮件是系统自动发送的。
        2.你必须将日报以邮件形式发送到系统指定的邮箱。
    """

    agent.invoke({
        "input": user_input
    })

if __name__ == "__main__":
    main()
