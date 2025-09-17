def run_in_manual_mode():
    from langserve import RemoteRunnable

    remote_chain = RemoteRunnable(url="http://localhost:8000/chat/")

    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    messages = [
        SystemMessage(content="你是一个摇滚老炮，你擅长根据用户表达从摇滚的角度回答用户的问题"),
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
    import json
    import time

    config = json.load(open('./config.json', 'r'))
    
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, trim_messages
    from langserve import RemoteRunnable

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def get_token_count(messages: list[BaseMessage]) -> int:
        total_tokens = 0
        for message in messages:
            message_str = f"{message.type}\n{message.content}"
            total_tokens += len(enc.encode(message_str))
            if hasattr(message, "name") and message.name:
                total_tokens += len(enc.encode(message.name))

        total_tokens += len(messages)

        return total_tokens

    trimmer = trim_messages(
        max_tokens=2000,
        strategy="last",
        token_counter=get_token_count,
        include_system=True,
        allow_partial=False
    )

    store = {}

    def get_message_history(session_id: str) -> BaseChatMessageHistory:
        return store[session_id]

    def set_message_history(session_id: str, messages: list[BaseMessage]) -> str:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
            store[session_id].add_messages([SystemMessage(content="你是一个摇滚老炮，你擅长根据用户表达从摇滚的角度回答用户的问题")])
        
        store[session_id].add_messages(messages)
        
        return session_id

    def display_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        # 可以通过这个函数观察trim_messages的效果
        # print(messages)
        return messages

    # 尝试集成trimmer
    remote_chain = (RunnableLambda(lambda input_msg, config: set_message_history(config["configurable"]["session_id"], input_msg)) 
        | RunnableLambda(lambda session_id: get_message_history(session_id).messages) 
        | trimmer 
        | RunnableLambda(lambda messages: display_messages(messages))
        | RemoteRunnable(url=config["ROBOT_BASE_URL"])
    )

    while True:
        user_input = input("请输入你的问题，输入exit退出：")
        if user_input == 'exit':
            break
        
        print("\n\n================response start================\n\n")
        response = ""
        # 尝试集成流式输出的功能
        for r in remote_chain.stream(
            [HumanMessage(content=f"{user_input}")],
            config={"configurable": {"session_id": config["SESSION_ID"]}}
        ):  
            print(r, end='')
            response += r
        
        print("\n\n================ response end ================\n\n")
        store[config["SESSION_ID"]].add_messages([AIMessage(content=response)])


if __name__ == '__main__':
    run_in_auto_mode()