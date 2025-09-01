# 命令行多功能智能体
import json
import inspect
from openai import OpenAI
from functools import wraps

from typing import Dict, Any, Callable, List, Optional
from typing import get_type_hints, get_origin, get_args

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
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
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
    
    def generate_reply(self, thought: str, observation: str) -> str:
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
        if len(response.get("choices")) > 0 and
            response.get("choices")[0].get("message", {}).get("content"):
            return response.get("choices")[0].get("message", {}).get("content")
        else:
            return "Agent 未能生成有效的回复。"
    
    def parse_action(self, llm_response: str) -> Optional[Dict[str, Any]]:
        response = json.loads(llm_response)
        if len(response.get("choices")) > 0 and
            response.get("choices")[0].get("message", {}).get("tool_calls"):
            action_wrapper = {
                "actions":response.get("choices")[0].get("message", {}).get("tool_calls"),
                "thought": response.get("choices")[0].get("finish_reason")
            }
            return action_wrapper
        else:
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