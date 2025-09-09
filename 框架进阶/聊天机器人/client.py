from langserve import RemoteRunnable

remote_chain = RemoteRunnable(url="http://localhost:8000/chat/")

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个mbti资深心理分析师，你擅长根据用户表达从mbti的角度分析用户的状态"),
]

while True:
    user_input = input("请输入你的问题，输入exit退出：")
    if user_input == "exit":
        break
    messages.append(HumanMessage(content=user_input))
    response = remote_chain.invoke(messages)
    messages.append(AIMessage(content=response))
    print(response， '\n')