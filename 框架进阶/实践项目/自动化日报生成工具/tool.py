from langchain_core.tools import tool
from typing import List
from custome_type import PaperInfo

import json

config = json.load(open("./config.json", encoding="utf-8"))

@tool
def count_arxiv_latest_papers(keywords: List[str]) -> int:
    """
    获取arxiv上指定关键词的当天发表的论文数量

    Args:
        keywords: 关键词列表

    Returns:
        论文数量
    """
    import arxiv
    from datetime import date, timedelta

    start_date = date.today() - timedelta(days=1)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date = date.today()
    end_date_str = end_date.strftime("%Y%m%d")

    query = " AND ".join(keywords)
    query += " AND submittedDate:[{} TO {}]".format(start_date_str, end_date_str)
    search = arxiv.Search(query=query)

    Client = arxiv.Client()
    count  = 0
    for result in Client.results(search):
        count += 1

    return count


@tool
def fetch_arxiv_latest_papers_with_limit(keywords: List[str], page: int = 0) -> List[PaperInfo]: 
    """
    获取arxiv上指定关键词和范围的当天发表的论文信息

    Args:
        keywords: 关键词列表
        page: 页码，第一页为0，每页最多10篇论文，默认为0

    Returns:
        论文信息列表，包含了每篇论文的标题、作者、摘要、提交日期
    """
    import arxiv
    from datetime import date, timedelta

    start_date = date.today() - timedelta(days=1)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date = date.today()
    end_date_str = end_date.strftime("%Y%m%d")

    results = []
    query = " AND ".join(keywords)
    query += " AND submittedDate:[{} TO {}]".format(start_date_str, end_date_str)
    search = arxiv.Search(
        query = query,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending
    )

    Client = arxiv.Client()
    for result in Client.results(search):
        results.append({
            "title": result.title,
            "authors": result.authors,
            "abstract": result.summary,
            "submitted_date": result.published.strftime("%Y-%m-%d")
        })

    return results[page * 10: (page + 1) * 10]

@tool
def send_email(subject: str, content: str) -> int:
    """
    发送邮件到系统指定的邮箱

    Args:
        subject: 邮件主题
        content: 邮件内容

    Returns:
        是否发送成功，成功返回0，失败返回1
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg['From'] = config["EMAIL"]["SENDER_EMAIL"]
    msg['To'] = ", ".join(config["EMAIL"]["RECIPIENT_EMAIL"])  # 处理多个收件人
    msg['Subject'] = subject

    msg.attach(MIMEText(content, 'plain', 'utf-8'))  # 添加编码
    try:
        # 使用SSL连接而不是TLS
        with smtplib.SMTP_SSL(config["EMAIL"]["SMTP_SERVER"], config["EMAIL"]["SMTP_PORT"]) as server:
            server.login(config["EMAIL"]["SENDER_EMAIL"], config["EMAIL"]["SENDER_PASSWORD"])
            server.send_message(msg)
            print("邮件发送成功")
        return 0
    except Exception as e:
        print(f"发送邮件时出错: {e}")
        return 1

tools = [
    count_arxiv_latest_papers,
    fetch_arxiv_latest_papers_with_limit,
    send_email
]
