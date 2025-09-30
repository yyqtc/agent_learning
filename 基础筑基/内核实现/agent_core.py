import json
import re
import logging

import inspect
from typing import Dict, Any, Callable, List, Optional
from typing import get_type_hints, get_origin, get_args
from functools import wraps

# ==================== 日志系统配置 ====================
# 配置全局日志格式：时间 [级别] 内容
# 输出到控制台，便于调试 Agent 的每一步行为
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO 及以上会被记录
    format='%(asctime)s [%(levelname)s] %(message)s',  # 时间 | 等级 | 消息
    handlers=[
        logging.StreamHandler()  # 输出到终端
    ]
)
logger = logging.getLogger(__name__)  # 创建一个独立的 logger 实例

# ==================== Agent 内核类 ====================
class Agent:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}  # 存储注册的工具函数，键为工具名
        self.tool_schemas: List[Dict] = []   # 存储每个工具的 JSON Schema 描述
        self.max_retries = 3                 # 工具调用失败时的最大重试次数
        
        config = json.load(open("config.json", "r"))
        self.qwen_api_key = config["QWen-API-KEY"]
        self.QWEN_API_BASE = config["QWEN_API_BASE"]

        self.messages = [
          {"role": "system", "content": "你是一个智能助手，请根据用户的问题，使用工具回答问题。"}
        ]


    def register_tool(self, func: Callable):
        """
        装饰器：用于注册工具函数，并自动生成其 JSON Schema
        使用方式：@agent.register_tool
        """
        # 构建该工具的标准描述结构（模仿 OpenAI Functions 格式）

        sig = inspect.signature(func)
        hints = get_type_hints(func)

        properties = {}
        required = []
        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
            tuple: "array",
            set: "array"
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
            "name": func.__name__,  # 工具函数名
            "description": (func.__doc__ or "").strip(),  # 函数的文档字符串作为描述
            "parameters": {
                "type": "object",
                "properties": properties,   # 参数属性（此处简化，实际可解析 type hints）
                "required": required      # 必填参数（可扩展）
            }
        }

        # 将工具存入字典，便于后续调用
        self.tools[func.__name__] = func
        # 将工具的 schema 加入列表，可用于提示 LLM 哪些工具可用
        self.tool_schemas.append(schema)

        # 保留原函数的元信息（如 __name__, __doc__）
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 输出的文本中解析出 Thought 和 Action 部分
        输入格式示例：
            Thought: 我需要计算 3+5
            Action: {"tool": "add", "parameters": {"a": 3, "b": 5}}
            Observation: ...
        """
        try:
            # 使用正则提取 "Thought:" 后的内容
            thought_match = re.search(r"Thought:\s*(.*?)\nAction:", text, re.DOTALL)
            # 提取 "Action:" 和 "Observation:" 之间的 JSON 字符串
            action_match = re.search(r"Action:\s*({.*?})\s*Observation:", text, re.DOTALL | re.MULTILINE)

            # 获取 Thought 内容，若未匹配则设为默认值
            thought = thought_match.group(1).strip() if thought_match else "No thought."
            # 获取 Action JSON 字符串
            action_json_str = action_match.group(1).strip() if action_match else None

            if not action_json_str:
                logger.warning("未在输出中找到有效的 Action JSON。")
                return None

            # 将 JSON 字符串解析为 Python 字典
            action = json.loads(action_json_str)
            # 将 Thought 也加入 action 字典，便于后续日志记录
            action["thought"] = thought
            return action  # 返回包含 thought 和 action 信息的字典

        except Exception as e:
            logger.error(f"解析 Action 失败：{e}")
            return None

    def _execute_action(self, action: Dict[str, Any]) -> str:
        """
        执行解析出的 Action，调用对应工具函数，支持失败重试
        返回 Observation（执行结果或错误信息）
        """
        tool_name = action.get("tool")        # 要调用的工具名
        tool_input = action.get("parameters", {})  # 工具的输入参数

        # 检查工具是否存在
        if tool_name not in self.tools:
            return f"Error: 工具 '{tool_name}' 未注册或不存在。"

        tool_func = self.tools[tool_name]  # 获取工具函数对象

        # 最多重试 max_retries 次
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"正在执行工具: {tool_name} (第 {attempt} 次尝试)")
                result = tool_func(**tool_input)  # 调用工具函数，传入参数
                # 成功则返回标准格式的 success 结果
                return json.dumps({"status": "success", "result": result})
            except Exception as e:
                # 记录错误日志
                logger.error(f"工具 {tool_name} 第 {attempt} 次执行失败: {str(e)}")
                # 如果是最后一次尝试，返回错误信息
                if attempt == self.max_retries:
                    return json.dumps({"status": "error", "message": str(e)})
                continue  # 继续下一次重试
        # 理论上不会走到这里，但防止异常
        return "Unknown execution error."

    def run(self, user_input: str):
        """
        Agent 主运行流程：
        1. 模拟 LLM 生成 Thought + Action
        2. 解析出 Action
        3. 执行工具并获取 Observation
        4. 生成最终回复
        """
        # 模拟大模型输出（实际中应替换为调用 Qwen 等 API）
        fake_llm_response = self._mock_llm_response(user_input)
        logger.info(f"LLM 输出内容:\n{fake_llm_response}")

        # 解析出 Thought 和 Action
        action = self._parse_action(fake_llm_response)
        if not action:
            return "Agent 未能生成有效的 Action。"

        thought = action["thought"]
        logger.info(f"🧠 推理过程 (Thought): {thought}")

        # 执行工具调用，获取观察结果
        observation = self._execute_action(action)
        logger.info(f"👀 执行反馈 (Observation): {observation}")

        # 模拟生成最终回复（实际中可让 LLM 根据 observation 总结）
        final_reply = self._mock_generate_reply(thought, observation)
        logger.info(f"💬 最终回复: {final_reply}")
        return final_reply

    def _mock_llm_response(self, user_input: str) -> str:
        """
        模拟大模型的输出（用于测试）
        实际项目中应替换为真实 API 调用（如 Qwen）
        """
        # 根据输入决定调用哪个工具（简化逻辑）
        if "天气" in user_input.lower():
            return '''Thought: 用户想知道北京的天气，我应该调用天气查询工具。
Action: {"tool": "get_weather", "parameters": {"city": "Beijing"}}
Observation: '''
        else:
            return '''Thought: 用户想计算 3+5，我应该使用计算器。
Action: {"tool": "add", "parameters": {"a": 3, "b": 5}}
Observation: '''


    def _mock_generate_reply(self, thought: str, observation: str) -> str:
        """
        模拟最终回复生成（实际中可让 LLM 总结）
        这里简单解析 observation 并返回结果
        """
        try:
            obs = json.loads(observation)  # 解析 observation JSON
            if obs["status"] == "success":
                result = obs["result"]
                return f"✅ 操作成功，结果是：{result}"
            else:
                msg = obs["message"]
                return f"❌ 操作失败：{msg}"
        except:
            return "⚠️ 无法解析执行结果。"
    

# ==================== 注册示例工具 ====================
# 创建 Agent 实例
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
    conditions = ["晴天", "多云", "小雨"]
    return f"{city} 今天气温 {random.choice(temps)}°C，{random.choice(conditions)}"

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("🔍 Agent 已启动，支持的工具列表:")
    for schema in agent.tool_schemas:
        print(f"  🛠️  {schema['name']}: {schema['description']}")

    print("\n🚀 开始运行 Agent...")
    # 测试加法
    agent.run("3 加 5 等于多少？")
    print("-" * 50)
    # 测试天气
    agent.run("北京今天天气怎么样？")