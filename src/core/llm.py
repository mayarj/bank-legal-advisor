from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import settings

llm = ChatAnthropic(
    model=settings.claude_model,
    api_key=settings.anthropic_api_key,
    temperature=settings.claude_temperature,
    max_tokens=settings.claude_max_tokens,
)


def invoke(system_msg: str, prompt: str) -> str:
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content
