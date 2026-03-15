from pathlib import Path
from typing import Any
import os
import time
import hashlib
import json


class ModelProvider:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.name = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.redis_url = os.getenv("REDIS_URL", "").strip()
        self._gemini = None
        self._openai = None
        self._redis = None
        self._cache = {}
        self._trace = []
        self._langfuse = None
        self._setup()

    def _setup(self):
        if self.redis_url:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url)
            except Exception:
                self._redis = None

        try:
            from langfuse import Langfuse
            self._langfuse = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST"),
            )
        except Exception:
            self._langfuse = None

        if self.gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._gemini = ChatGoogleGenerativeAI(
                    model=self.gemini_model,
                    google_api_key=self.gemini_key,
                    temperature=0
                )
            except Exception:
                self._gemini = None

        if self.openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self._openai = ChatOpenAI(
                    model=self.openai_model,
                    api_key=self.openai_key,
                    temperature=0
                )
            except Exception:
                self._openai = None

    @property
    def llm(self):
        if self.name == "gemini" and self._gemini:
            return self._gemini

        if self.name == "openai" and self._openai:
            return self._openai

        if self._gemini:
            return self._gemini

        if self._openai:
            return self._openai

        return None

    def _cache_key(self, question: str, context: list[dict[str, Any]], intent: str, sentiment: str):
        joined = "|".join([item["answer"] for item in context])
        raw = f"{question}|{joined}|{intent}|{sentiment}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cache(self, key):
        if self._redis:
            try:
                value = self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception:
                pass
        return self._cache.get(key)

    def _set_cache(self, key, value):
        if self._redis:
            try:
                self._redis.setex(key, 3600, json.dumps(value))
            except Exception:
                pass
        self._cache[key] = value

    def _tool_router(self, question: str, intent: str):
        q = question.lower()

        if "order" in q or "track" in q:
            return "order_status"

        if "refund" in q or "return" in q:
            return "refund_policy"

        if "ship" in q or "delivery" in q:
            return "shipping_policy"

        if intent.lower() in ["login", "account", "password"]:
            return "faq"

        return "llm"

    def _execute_tool(self, tool, context):
        if tool == "faq" and context:
            return context[0]["answer"]

        if tool == "order_status":
            return "You can track your order from the Orders page in your account dashboard."

        if tool == "refund_policy":
            return "Refunds are processed within 3–5 business days after the returned item is received."

        if tool == "shipping_policy":
            return "Most orders ship within 24 hours and arrive in 5–7 business days."

        return None

    def _build_prompt(self, question, context, intent, sentiment, tool):
        joined = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in context])

        triage = (
            "You are a support triage agent.\n"
            f"Intent detected: {intent}\n"
            f"User sentiment: {sentiment}\n"
            f"Selected tool: {tool}\n"
        )

        retrieval = (
            "You are a knowledge retrieval specialist.\n"
            "Use the support knowledge base context if relevant.\n"
            f"{joined}\n"
        )

        resolution = (
            "You are a service desk resolution specialist.\n"
            "Provide a short and clear answer.\n"
            f"User question:\n{question}"
        )

        return f"{triage}\n{retrieval}\n{resolution}"

    def _invoke_gemini(self, prompt):
        response = self._gemini.invoke(prompt)
        return getattr(response, "content", str(response))

    def _invoke_openai(self, prompt):
        response = self._openai.invoke(prompt)
        return getattr(response, "content", str(response))

    def generate(self, question: str, context: list[dict[str, Any]], intent: str, sentiment: str):
        self._trace = []

        key = self._cache_key(question, context, intent, sentiment)

        cached = self._get_cache(key)
        if cached:
            self._trace.append("cache_hit")
            return cached, self._trace

        tool = self._tool_router(question, intent)
        self._trace.append(f"tool_router:{tool}")

        tool_result = self._execute_tool(tool, context)

        if tool_result:
            self._trace.append("tool_execution")
            self._set_cache(key, tool_result)
            return tool_result, self._trace

        prompt = self._build_prompt(question, context, intent, sentiment, tool)

        if self._gemini:
            try:
                self._trace.append("llm:gemini")
                result = self._invoke_gemini(prompt)
                self._set_cache(key, result)
                return result, self._trace
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(2)
                    try:
                        result = self._invoke_gemini(prompt)
                        self._set_cache(key, result)
                        return result, self._trace
                    except Exception:
                        pass

        if self._openai:
            try:
                self._trace.append("llm:openai")
                result = self._invoke_openai(prompt)
                self._set_cache(key, result)
                return result, self._trace
            except Exception:
                pass

        if context:
            self._trace.append("fallback:faq")
            result = context[0]["answer"]
            self._set_cache(key, result)
            return result, self._trace

        self._trace.append("fallback:none")
        result = "I could not find a specific policy match in the knowledge base."
        self._set_cache(key, result)
        return result, self._trace