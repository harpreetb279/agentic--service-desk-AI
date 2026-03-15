import json
import logging
import os
import time
from typing import Any


class Monitor:
    def __init__(self):
        self.logger = logging.getLogger("servicedesk-ai")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

        self.langfuse_enabled = False
        self.client = None
        self.debug_info = {
            "enable_langfuse_env": os.getenv("ENABLE_LANGFUSE", ""),
            "public_key_present": False,
            "secret_key_present": False,
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "init_error": None,
        }

        enabled = os.getenv("ENABLE_LANGFUSE", "false").strip().lower() == "true"
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

        self.debug_info["public_key_present"] = bool(public_key)
        self.debug_info["secret_key_present"] = bool(secret_key)
        self.debug_info["host"] = host

        if not enabled:
            self.debug_info["init_error"] = "ENABLE_LANGFUSE is not true"
            return

        if not public_key:
            self.debug_info["init_error"] = "LANGFUSE_PUBLIC_KEY is missing"
            return

        if not secret_key:
            self.debug_info["init_error"] = "LANGFUSE_SECRET_KEY is missing"
            return

        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self.langfuse_enabled = True
        except Exception as e:
            self.client = None
            self.langfuse_enabled = False
            self.debug_info["init_error"] = str(e)

    def start(self, request_id: str, question: str):
        started = time.time()
        self.logger.info(json.dumps({
            "event": "request_started",
            "request_id": request_id,
            "question": question,
        }))
        trace = None
        if self.langfuse_enabled and self.client is not None:
            try:
                trace = self.client.trace(
                    name="servicedesk_request",
                    id=request_id,
                    input={"question": question},
                )
            except Exception as e:
                self.debug_info["init_error"] = f"trace_start_failed: {e}"
                trace = None
        return started, trace

    def finish(self, request_id: str, started: float, result: dict[str, Any], trace):
        latency_ms = int((time.time() - started) * 1000)
        self.logger.info(json.dumps({
            "event": "request_finished",
            "request_id": request_id,
            "latency_ms": latency_ms,
            "intent": result.get("intent"),
            "provider": result.get("provider"),
            "tools_used": result.get("tools_used", []),
            "status": result.get("status"),
        }))
        if self.langfuse_enabled and trace is not None:
            try:
                trace.update(
                    output=result,
                    metadata={"latency_ms": latency_ms},
                )
            except Exception as e:
                self.debug_info["init_error"] = f"trace_finish_failed: {e}"
        result["latency_ms"] = latency_ms
        return result