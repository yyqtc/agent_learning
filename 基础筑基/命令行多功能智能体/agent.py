# 命令行多功能智能体
import json
import inspect
from openai import OpenAI
from functools import wraps
import logging

from typing import Dict, Any, Callable, List, Optional
from typing import get_type_hints, get_origin, get_args

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
        config = json.load(open("config.json", "r"))
        self.qwen_api_key = config["QWen-API-KEY"]
        self.qwen_api_base = config["QWen-API-BASE"]

        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict] = []

        self.max_retries = 3

        self.messages = [
            {"role": "system", "content": "你是一个智能助手，请根据用户的问题，使用工具回答问题。"}
        ]

    def resume_beggining_messages(self):
        self.messages = [
            {"role": "system", "content": "你是一个智能助手，请根据用户的问题，使用工具回答问题。"}
        ]

    def register_tool(self, func: Callable):
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties = {}
        required = []

        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            dict: "object",
            list: "array",
            bool: "boolean"
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            
            hint = hints.get(name)
            type = type_map.get(hint, "string")
            properties[name] = {"type": type, "description": f"参数{name}的类型是{type}"}
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
    
    def llm_response(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        client = OpenAI(
            api_key=self.qwen_api_key,
            base_url=self.qwen_api_base
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=self.messages,
            tools=self.tool_schemas
        )
        return completion.model_dump_json()
    
    def generate_reply(self, thought: str, observation: str) -> Dict[str, Any]:
        self.messages.append({
            "role": "assistant", 
            "content": f'大模型的选择是：{thought}，工具执行结果是：{observation}，请隐藏思考过程，直接输出最终答案'
        })
        client = OpenAI(
            api_key=self.qwen_api_key,
            base_url=self.qwen_api_base
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=self.messages
        )
        response = json.loads(completion.model_dump_json())
        if len(response.get("choices")) > 0:
            if response.get("choices")[0].get("finish_reason") == "tool_calls":
                return {
                    "status": "tool_calls",
                    "result": completion.model_dump_json()
                }
            elif response.get("choices")[0].get("finish_reason") == "stop":
                if response.get("choices")[0].get("message", {}).get("content"):
                    return {
                        "status": "final_reply",
                        "result": response.get("choices")[0].get("message", {}).get("content")
                    }
                else:
                    return {"result": "Agent 未能生成有效的回复。", "status": "fail"}

            else:
                return {"result": "Agent 未能生成有效的回复。", "status": "fail"}

        else:
            return {"result": "Agent 未能生成有效的回复。", "status": "fail"}
    
    def parse_action(self, llm_response: str) -> Optional[Dict[str, Any]]:
        try:
            print(llm_response)
            response = json.loads(llm_response)
            if (len(response.get("choices")) > 0 and
                response.get("choices")[0].get("message", {}).get("tool_calls")):
                action_wrapper = {
                    "actions":response.get("choices")[0].get("message", {}).get("tool_calls"),
                    "thought": response.get("choices")[0].get("finish_reason")
                }
                return action_wrapper
            else:
                return None
        except Exception as e:
            logger.error(f"解析大模型response失败{e}", )
            return None
    
    def execute_action(self, action: Dict[str, Any]) -> str:
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
                        result[name] = {
                            "status": "success",
                            "result": tool_func(**args)
                        }
                        break
                    except Exception as e:
                        logger.error(f"工具 {name} 第 {attempt} 次执行失败: {str(e)}")

            except Exception as e:
                logger.error(f"执行动作的过程中发生了以下问题{e}")
                result[name] = {
                    "status": "fail",
                    "result": e
                }

        return json.dumps(result)

    def run(self, user_input: str):
        logger.info(f"用户输入：{user_input}")

        self.messages.append({"role": "user", "content": user_input})

        llm_response = self.llm_response(user_input)
        action = self.parse_action(llm_response)
        if not action:
            logger.info("Agent 未能生成有效的 Action。")
            return "Agent 未能生成有效的回复。"

        while True:
            thought = action["thought"]
            logger.info(f"🧠 推理过程 (Thought): {thought}")
            observation = self.execute_action(action)
            logger.info(f"👀 执行反馈 (Observation): {observation}")
            response = self.generate_reply(thought, observation)
            if response["status"] == "tool_calls":
                action = self.parse_action(response["result"])
            elif response["status"] == "final_reply":
                self.resume_beggining_messages()
                logger.info(f"💬 最终回复: {response['result']}")
                return response["result"]
            else:
                self.resume_beggining_messages()
                logger.info("💬 最终回复: Agent 未能生成有效的回复。")
                return "Agent 未能生成有效的回复。"


