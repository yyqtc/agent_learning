from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily

import json
import asyncio

config = json.load(open("./config.json", "r", encoding="utf-8"))

async def main():
    model_info = {
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.R1,
        "structured_output": True
    }

    client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=config["DEEPSEEK_API_KEY"],
        base_url=config["DEEPSEEK_API_BASE"],
        model_info=model_info,
        temperature=0.7,
        max_tokens=2000
    )

    assistant = AssistantAgent(name="assistant", model_client=client)
    user_proxy = UserProxyAgent(name="user")

    termination_condition = TextMentionTermination("TERMINATE") | MaxMessageTermination(2)

    group_chat = RoundRobinGroupChat(
        [assistant, user_proxy],
        termination_condition=termination_condition
    )

    stream = group_chat.run_stream(task="hello, world!")
    await Console(stream)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())