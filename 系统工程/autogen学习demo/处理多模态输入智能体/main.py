from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily

from io import BytesIO
from PIL import Image

import requests
import asyncio
import json

config = json.load(open("./config.json", "r", encoding="utf-8"))

async def main():
    model_info = {
        "vision": True,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.R1,
        "structured_output": False
    }

    model_client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=config["DEEPSEEK_API_KEY"],
        base_url=config["DEEPSEEK_API_BASE"],
        model_info=model_info,
        temperature=0.7,
        max_tokens=2000
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client
    )

    pil_image = Image.open(
        BytesIO(
            requests.get("https://picx.zhimg.com/80/v2-cc4c696a2c09d47c0835f29f97c9a6be_1440w").content
        )
    )
    
    # 将PIL图像转换为autogen的Image对象
    from autogen_core import Image as AGImage
    ag_image = AGImage(pil_image)
    
    multi_modal_message = MultiModalMessage(
        content=[
            "能以一位迷幻摇滚老炮的视角描述一下这张图吗",
            ag_image
        ],
        source="user"
    )

    stream = agent.run_stream(task=multi_modal_message)
    await Console(stream)

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
