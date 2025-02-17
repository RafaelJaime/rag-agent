from os import getenv
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

class LLMModels(Enum):
    """Available LLM models."""
    OPENAI_GPT4OMINI = "openai/gpt-4o-mini"
    OPENAI_GPTO3MINI = "openai/o3-mini"
    DEEPSEEK="deepseek/deepseek-r1:free"
    DEEPSEEK_15 = "deepseek-r1:1.5b"
    DEEPSEEK_70b = "deepseek/deepseek-r1-distill-llama-70b:free" # Free on open router

class LLMurls(Enum):
    """URLs for accessing LLM models."""
    LOCAL_OLLAMA_URL = "http://localhost:11434"
    OPENROUTER_URL = getenv("OPENROUTER_BASE_URL")

class LLM:
    @staticmethod
    def chatOpenAIWithOpenRouter(model: LLMModels) -> ChatOpenAI:
        """
        Create a chat with openai_api_key - Openrouter With gpt-4o

        :return: ChatOpenAI from Langchain_community library.
        :rtype: ChatOpenAI
        """
        return ChatOpenAI(
                openai_api_key=getenv("OPENROUTER_API_KEY"),
                openai_api_base=LLMurls.OPENROUTER_URL.value,
                model_name=model.value,
                model_kwargs={
                    "extra_headers": {
                        "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
                    }
                },
            )

    @staticmethod
    def chatWithOllama(model: LLMModels) -> ChatOllama:
        """
        Create a chat with Ollama running locally.

        :return: ChatOllama instance from Langchain_community library.
        :rtype: ChatOllama
        """
        return ChatOllama(
            model=model.value,
            base_url=LLMurls.LOCAL_OLLAMA_URL.value,
            model_kwargs={
                "extra_headers": {
                    "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
                }
            },
        )

