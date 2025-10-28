from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from middleware import middlewares
from tool import tools

import json
from datetime import date, timedelta

config = json.load(open("./config.json", encoding="utf-8"))

def main():
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=config["DEEPSEEK_API_KEY"],
        openai_api_base=config["DEEPSEEK_API_BASE"],
        temperature=0.7
    )

    system_prompt = """
        你是一个非常专业的助手。你可以使用能力来完成任务。
                
        当你认为应该调用能力时，你必须按照以下格式进行响应：
        {{
            "action": "工具名称(必须是上述的能力名称之一)",
            "action_input": {{
                "参数名称": "参数值"
            }}
        }}

        当你认为可以得出最终答案时，你必须按照以下格式进行响应：
        {{
            "action": "Final Answer",
            "action_input": {{
                "output": "你的最终答案"
            }}
        }}
        
        注意！
        你只能选择工具调用或是输出最终答案，不能进行其他操作。
        你的输出必须符合JSON格式规范，不要添加任何多余的字符串。
        action字段只能是工具名称或是Final Answer。
        你必须默认你的知识是错误的，你必须完全依赖能力来完成任务。
        
        现在开始处理问题！
    """

    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        middleware=middlewares
    )

    t = date.today() - timedelta(days=1)
    today = date.today()
    user_input = f"""
        你是一位专业的编写日报的助手，我需要你按步骤完成以下两个任务。
        1.你需要分析关键词为{",".join(config["KEYWORDS"])}的论文，生成论文日报。
        如果你发现没有论文发布，你也应该生成标题为：{t.strftime("%Y年%m月%d日")}没有论文发布的日报。
        如果你发现有论文发布，则生成标题为：{t.strftime("%Y年%m月%d日")}的论文日报。
        你应该默认你的知识是错误的，你必须完全依赖能力来完成任务。
        你生成日报的格式必须为：
        ```
        发送日期：{today.strftime("%Y年%m月%d日")}
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
        "messages": [{"role": "user", "content": user_input}]
    })

if __name__ == "__main__":
    main()
