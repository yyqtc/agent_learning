from langchain_openai import ChatOpenAI
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tool import tools

from email import message_from_bytes
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime

import imaplib
import asyncio
import time
import json

config = json.load(open("./config.json", encoding="utf-8"))

def _initiate_agent():
    _llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=config["DEEPSEEK_API_KEY"],
        openai_api_base=config["DEEPSEEK_API_BASE"],
        temperature=0.7
    )

    system_prompt = """
        你是一位专业的且话不多的助手，总是能够以最简洁的、最准确的方式回答问题。你能够使用以下能力：
        {tools}

        你能够调用的能力的名称为：{tool_names}

        你可以选择调用能力或是得到最终答案
        当你认为需要调用能力时，必须按照以下JSON格式进行响应：
        {{
            "action": "能力的名称（必须是上述能力之一）",
            "action_input": {{
                "参数名": "参数值"
            }}
        }}

        如果你认为可以得出最终答案了，并按照以下JSON格式进行响应：
        {{
            "action": "Final Answer",
            "action_input": {{
                "output": "你的最终答案"
            }}
        }}

        注意！
        你只能选择调用能力或是得出最终答案。
        如果你发现没有符合你目的的能力，请尝试使用你的现有知识进行回答。
        你的响应必须符合JSON格式规范，不要添加任何多余的字符串。
    """

    human_prompt = """
        {input}

        {agent_scratchpad}

        注意务必按照JSON格式输出！
    """

    _prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    agent = create_structured_chat_agent(
        llm=_llm,
        tools=tools,
        prompt=_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=100,
        handle_parsing_errors=True
    )

    return agent_executor

def _get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                continue
            elif content_type == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                body += part.get_payload(decode=True).decode(charset, errors="ignore")
            elif content_type == "text/html":
                continue
    else:
        charset = msg.get_content_charset() or "utf-8"
        body += msg.get_payload(decode=True).decode(charset, errors="ignore")

    return body

def main():
    agent_executor = _initiate_agent()

    imaplib.Commands["ID"]=("AUTH")
    mail_server = imaplib.IMAP4_SSL(config["EMAIL"]["IMAP"]["SERVER"], config["EMAIL"]["IMAP"]["PORT"])
    mail_server.login(config["EMAIL"]["SENDER_EMAIL"], config["EMAIL"]["SENDER_PASSWORD"])

    # RFC 2971 导致必须进行二次验证
    args = ("name","15108264418","contact","15108264418@163.com","version","1.0.0","vendor","myclient")
    typ, dat = mail_server._simple_command('ID', '("' + '" "'.join(args) + '")')

    mail_server.select("INBOX")

    print("开始监听邮箱...")
    status, email_ids = mail_server.search(None, "UNSEEN")
    if not email_ids or len(email_ids) == 0:
        time.sleep(10)
        return

    checked_email_ids = []
    for email_id in email_ids:
        if len(email_id) == 0:
            continue

        decoded_email_id = email_id
        if type(email_id) == bytes:
            decoded_email_id = email_id.decode()

        if " " in decoded_email_id:
            splitted_email_ids = decoded_email_id.split(" ")
            for id in splitted_email_ids:
                checked_email_ids.append(id.encode())

        else:
            checked_email_ids.append(decoded_email_id.encode())

    for email_id in checked_email_ids:
               
        print(f"开始处理邮件：{email_id}")

        status, email_data = mail_server.fetch(email_id, "(RFC822)")
        raw_email = email_data[0][1]
        msg = message_from_bytes(raw_email)

        subject = decode_header(msg.get("Subject", ""))[0][0].decode("utf-8")
        if subject != config["EMAIL"]["SUBJECT"]:
            continue
            
        from_addr = parseaddr(msg.get("From", ""))[1]
        email_body = _get_email_body(msg)
            
        prompt = f"""
            你是一位非常专业细致并且话不多的俄语老师，我需要你检查俄语单词背诵情况，并将检查结果发送给地址为{from_addr}的邮箱。
            你首先应该检查邮件内容，如果邮件内容和俄语单词无关，则直接回复“和俄语单词背诵无关的问题老夫不回答”给地址为{from_addr}的邮箱。
            如果邮件内容和俄语单词有关，则默认邮件内容遵循以下格式：
            如果是名词，则格式为：
            中文意思 单词单数形式（如果单词只有复数形式，则省略单数形式） 单词复数形式（复数形式可省略，但是如果存在则应该一起检查）
            如果是动词、形容词、代词、数词、副词、其他虚词或是短语，则格式为：
            中文意思 单词（或短语）

            用户的邮件内容为：
            {email_body}

            你需要逐行检查背诵情况，并在每行后面添加检查结果，最后你输出的检查结果应该遵循以下格式：
            如果是名词，且背诵情况为正确：
            中文意思 单词单数形式 单词复数形式 ✔
            如果是名词，且背诵情况为错误：
            中文意思 单词单数形式 单词复数形式 ❌ 错误原因
            如果是动词、形容词、代词、数词、副词、其他虚词或是短语，且背诵情况为正确：
            中文意思 单词（或短语） ✔
            如果是动词、形容词、代词、数词、副词、其他虚词或是短语，且背诵情况为错误：
            中文意思 单词（或短语） ❌ 错误原因

            注意！
            你必须调用能力发送邮件，而不应该模拟邮件发送！
            邮件只能发送一次，请不要反复发送邮件骚扰用户！
            邮件主题为：俄语单词背诵检查结果
            邮件内容为你输出的检查结果
        """

        asyncio.run(agent_executor.ainvoke({"input": prompt}))
        mail_server.store(email_id, "+FLAGS", "\\Seen")
        

if __name__ == "__main__":
    main()
