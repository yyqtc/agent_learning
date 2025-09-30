import json
import logging

import inspect
from typing import Dict, Any, Callable, List, Optional
from typing import get_type_hints, get_origin, get_args
from functools import wraps

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict] = []

        self.max_retries = 3

        self.messages: List[Dict] = [
            {"role": "system", "content": "你是一个智能助手，请根据用户的问题，使用工具回答问题。"}
        ]

        config = json.load(open("config.json", "r"))
        self.qwen_api_key = config["QWen-API-KEY"]
        self.QWEN_API_BASE = config["QWEN_API_BASE"]   

    def register_tool(self, func: Callable):
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties = {}
        required = []
        
        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array"
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            hint = hints.get(name)
            json_type = type_map.get(hint, "string")
            properties[name] = {"type": json_type, "description": f"参数{name}的类型是{json_type}"}
            if param.default == inspect.Parameter.empty:
                required.append(name)
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": (func.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

        self.tools[func.__name__] = func
        self.tool_schemas.append(schema)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def _qwen_llm_response(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        client = OpenAI(
            api_key=self.qwen_api_key,
            base_url=self.QWEN_API_BASE
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=self.messages,
            tools=self.tool_schemas
        )
        return completion.model_dump_json()

    def _qwen_generate_reply(self, thought: str, observation: str) -> str:
        self.messages.append({"role": "assistant", "content": f'大模型的选择是：{thought}，工具执行结果是：{observation}，请总结并输出最终答案'})
        
        client = OpenAI(
            api_key=self.qwen_api_key,
            base_url=self.QWEN_API_BASE
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=self.messages
        )
        response = json.loads(completion.model_dump_json())
        return response.get("choices")[0].get("message", {}).get("content")

    def run(self, user_input: str):
        logger.info(f"💬 用户输入: {user_input}")

        llm_response = self._qwen_llm_response(user_input)

        action = self._parse_action(llm_response)
        final_reply = ""
        if not action:
            final_reply = "Agent 未能生成有效的 Action。"
        else:
            thought = action["thought"]
            logger.info(f"🧠 推理过程 (Thought): {thought}")

            observation = self._execute_action(action)
            logger.info(f"👀 执行反馈 (Observation): {observation}")

            final_reply = self._qwen_generate_reply(thought, observation)
            logger.info(f"💬 最终回复: {final_reply}")

        self.messages = [
            {"role": "system", "content": "你是一个智能助手，请根据用户的问题，使用工具回答问题。"}
        ]

        return final_reply

    def _parse_action(self, llm_response: str) -> Optional[Dict[str, Any]]:
        response = json.loads(llm_response)
        if (response.get("choices") \
            and len(response.get("choices")) \
            and response.get("choices")[0].get("message", {}).get("tool_calls")):
            actions_wrapper = {
                "actions": response.get("choices")[0].get("message", {}).get("tool_calls")
            }
            actions_wrapper["thought"] = response.get("choices")[0].get("finish_reason")
            return actions_wrapper
        else:
            return None

    def _execute_action(self, action: Dict[str, Any]) -> str:
        result = {}
        for action in action["actions"]:
            try:
                name = action.get("function", {}).get("name", "")
                args = json.loads(action.get("function", {}).get("arguments"))
                if name not in self.tools:
                    raise ValueError(f"执行工具{name}未被注册")

                tool_func = self.tools[name]
                for attempt in range(1, self.max_retries + 1):
                    try:
                        logger.info(f"正在执行工具: {name} (第 {attempt} 次尝试)")
                        result[name] = {
                            "status": "success",
                            "result": tool_func(**args)
                        }
                        break

                    except Exception as e:
                        logger.error(f"工具 {name} 第 {attempt} 次执行失败: {str(e)}")
                        if attempt == self.max_retries:
                            result[name] = str(e)
                            break
                        continue
            except Exception as e:
                logger.error(str(e))
                result[name] = str(e)

        return json.dumps(result)

agent = Agent()

@agent.register_tool
def add(a: int, b: int) -> int:
    """
    获取两个整数相加的结果。
    """
    return a + b

@agent.register_tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气。
    """
    import random
    temps = [20, 22, 25, 27, 30]
    return f"今天{city}的天气是{random.choice(temps)}度。"


if __name__ == "__main__":
    for schema in agent.tool_schemas:
        print(f"  🛠️  {schema['function']['name']}: {schema['function']['description']}")

    agent.run("3 加 5 等于多少？")
    agent.run("北京今天天气怎么样？")
