"""LM Studio model for LangChain - importable module."""
from langchain_openai import ChatOpenAI

llm_studio_model = ChatOpenAI(
    model="local-model",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.7,
    max_tokens=512,
)

