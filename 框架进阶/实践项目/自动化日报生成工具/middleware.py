from langchain.agents.middleware import before_model
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from typing import Any, Dict

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@before_model
def get_state_info(state: AgentState, runtime: Runtime) -> Dict[str, Any]:
    logger.info(f'messages: {state["messages"]}')
    return None

middlewares = [
    get_state_info
]

