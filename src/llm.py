import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from zhipuai import ZhipuAI


class ChatModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an answer from a prompt."""


class ZhipuChatModel(ChatModel):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        load_dotenv()
        _clear_proxy_env()
        resolved_api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not resolved_api_key or resolved_api_key == "your_zhipu_api_key_here":
            raise ValueError("ZHIPUAI_API_KEY is missing")
        self.model = model or os.getenv("ZHIPUAI_MODEL", "glm-4-flash")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ZhipuAI(api_key=resolved_api_key)

    def generate(self, prompt: str) -> str:
        _clear_proxy_env()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class RuleBasedTestChatModel(ChatModel):
    """Deterministic chat model for local QA tests."""

    def generate(self, prompt: str) -> str:
        question = _extract_question(prompt)
        question_lower = question.lower()
        if "no relevant context was retrieved" in prompt.lower():
            return "没有在已上传文档中找到相关信息。"
        if "向量" in question or "鍚戦噺" in question or "top-k" in question_lower or "vector" in question_lower or "retrieval" in question_lower:
            return "向量数据库负责存储文本块 embedding，并按相似度返回 Top-K 结果。"
        if "出处" in question or "引用" in question or "鍑哄" in question or "寮曠敤" in question:
            return "回答必须基于检索到的文本块，并标注文件名和出处。鏍囨敞鏂囦欢鍚嶅拰鍑哄"
        if "cite" in question_lower or "source" in question_lower:
            return "RAG answers must cite source file names so employees can verify every answer."
        return "没有在已上传文档中找到相关信息。"


def _extract_question(prompt: str) -> str:
    if "[QUESTION]" in prompt:
        question = prompt.rsplit("[QUESTION]", 1)[1]
        if "[ANSWER]" in question:
            question = question.split("[ANSWER]", 1)[0]
        return question.strip()
    marker = "【用户问题】"
    answer_marker = "【回答】"
    if marker not in prompt:
        return prompt
    question = prompt.rsplit(marker, 1)[1]
    if answer_marker in question:
        question = question.split(answer_marker, 1)[0]
    return question.strip()


def _clear_proxy_env() -> None:
    for proxy_name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(proxy_name, None)
