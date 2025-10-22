from langchain_core.tools import tool
import json

config = json.load(open("./config.json", encoding="utf-8"))

@tool
def send_email(to: str, subject: str, content: str) -> int:
    """
    发送邮件到邮箱

    Args:
        to: 邮件接收者邮箱地址
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
    msg['To'] = to 
    msg['Subject'] = subject

    msg.attach(MIMEText(content, 'plain', 'utf-8'))  # 添加编码
    try:
        # 使用SSL连接而不是TLS
        with smtplib.SMTP_SSL(config["EMAIL"]["SMTP"]["SERVER"], config["EMAIL"]["SMTP"]["PORT"]) as server:
            server.login(config["EMAIL"]["SENDER_EMAIL"], config["EMAIL"]["SENDER_PASSWORD"])
            server.send_message(msg)
        print(f"邮件发送成功：{to}")
        return 0
    except Exception as e:
        print(f"邮件发送失败：{to}")
        return 1

tools = [
    send_email
]