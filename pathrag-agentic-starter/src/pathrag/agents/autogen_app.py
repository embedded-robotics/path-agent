import os
from autogen import AssistantAgent, UserProxyAgent, tool
from typing import Dict, Any
from pathrag.agents.tools import prepare_and_run
from pathrag.utils.logging import get_logger

logger = get_logger("pathrag.autogen")

@tool
def pathrag(image_path: str, question: str, mode: str = "answer", top_k: int = 3) -> Dict[str, Any]:
    return prepare_and_run(image_path=image_path, question=question, mode=mode, top_k=top_k)

def build_agents():
    logger.info("Building AutoGen planner + user agents")
    planner = AssistantAgent(
        "planner",
        system_message="You plan Path-RAG steps and call tools.",
        llm_config={"config_list":[{"model":"gpt-4o-mini","api_key":os.getenv("OPENAI_API_KEY")}], "temperature":0},
        tools=[pathrag],
    )
    user = UserProxyAgent("user")
    return user, planner
