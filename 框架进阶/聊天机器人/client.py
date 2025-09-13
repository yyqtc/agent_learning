def run_in_manual_mode():
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
        print('\n', response, '\n')


def run_in_auto_mode():
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
    from langserve import RemoteRunnable
    import json

    config = json.load(open('./config.json', 'r'))

    from langchain_core.messages import HumanMessage, SystemMessage

    store = {}
    def get_message_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
            store[session_id].add_messages([SystemMessage(content="你是一个mbti资深心理分析师，你擅长根据用户表达从mbti的角度分析用户的状态")])
        return store[session_id]

    remote_chain = RunnableWithMessageHistory(
        RemoteRunnable(url=config["ROBOT_BASE_URL"]),
        get_message_history
    )

    while True:
        user_input = input("请输入你的问题，输入exit退出：")
        response = remote_chain.invoke(
            [HumanMessage(content=f"{user_input}")],
            config={"configurable": {"session_id": config["SESSION_ID"]}}
        )
        print('\n', response, '\n')



if __name__ == '__main__':
    run_in_auto_mode()