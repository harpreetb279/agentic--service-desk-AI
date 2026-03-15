from pathlib import Path
import importlib.util
import os
import uuid
from langchain_core.prompts import ChatPromptTemplate
from models.crew_mode import CrewCoordinator
from models.faq_retriever import FAQRetriever
from models.intent_model import IntentClassifier
from models.monitoring import Monitor
from models.providers import ModelProvider
from models.sentiment_model import SentimentClassifier
from notify import send_email_alert


class SupportAgent:

    def __init__(self, base_dir: Path):

        self.base_dir = Path(base_dir)

        self.intent_classifier = IntentClassifier(base_dir=self.base_dir)
        self.sentiment_classifier = SentimentClassifier(base_dir=self.base_dir)

        self.retriever = FAQRetriever(base_dir=self.base_dir)
        self.provider = ModelProvider(base_dir=self.base_dir)

        self.monitor = Monitor()
        self.crew = CrewCoordinator(self.provider)

        langgraph_flag = self._is_enabled("ENABLE_LANGGRAPH", True)
        crew_flag = self._is_enabled("ENABLE_CREWAI", True) or self._is_enabled("CREW_ENABLED", True)

        self.workflow_enabled = self._module_exists("langgraph") and langgraph_flag
        self.crew_enabled = self.crew.available and crew_flag

        self.feature_flags = {
            "langchain_installed": self._module_exists("langchain"),
            "langgraph_installed": self._module_exists("langgraph"),
            "crewai_installed": self._module_exists("crewai"),
            "langfuse_installed": self._module_exists("langfuse"),
            "openai_installed": self._module_exists("openai"),
            "gemini_installed": (
                self._module_exists("google.generativeai")
                or self._module_exists("google.genai")
            ),
        }

        self.prompt = ChatPromptTemplate.from_template(
            "Intent: {intent}\n"
            "Sentiment: {sentiment}\n"
            "Question: {question}\n"
            "Primary policy answer: {primary_answer}\n"
            "Respond in a concise helpful way."
        )

        self._graph = None

    def _module_exists(self, name: str):
        return importlib.util.find_spec(name) is not None

    def _is_enabled(self, key: str, default: bool):
        value = os.getenv(key, str(default)).strip().lower()
        return value == "true"

    def readiness(self):
        return {
            "provider": self.provider.name,
            "workflow_enabled": self.workflow_enabled,
            "crew_enabled": self.crew_enabled,
            "langfuse_enabled": self.monitor.langfuse_enabled,
            "vector_backend": self.retriever.backend_name,
            "feature_flags": self.feature_flags,
        }

    def get_faq_questions(self):
        return self.retriever.questions

    def debug_state(self):
        return {
            "provider": self.provider.name,
            "workflow_enabled": self.workflow_enabled,
            "crew_enabled": self.crew_enabled,
            "langfuse_enabled": self.monitor.langfuse_enabled,
            "vector_backend": self.retriever.backend_name,
            "faq_count": len(self.retriever.questions),
            "feature_flags": self.feature_flags,
        }

    def handle_query(self, question: str, mode: str = "auto", session_id: str | None = None):

        request_id = uuid.uuid4().hex[:12]
        started, trace = self.monitor.start(request_id, question)

        if mode == "crew" and self.crew_enabled:
            result = self._run_crew(request_id, question, session_id)

        elif mode == "graph" and self.workflow_enabled:
            result = self._run_graph(request_id, question, session_id)

        elif mode == "auto":
            if self.workflow_enabled:
                result = self._run_graph(request_id, question, session_id)
            elif self.crew_enabled:
                result = self._run_crew(request_id, question, session_id)
            else:
                result = self._run_linear(request_id, question, session_id)

        else:
            result = self._run_linear(request_id, question, session_id)

        return self.monitor.finish(request_id, started, result, trace)

    def _base_state(self, request_id: str, question: str, session_id: str | None):

        return {
            "request_id": request_id,
            "session_id": session_id,
            "question": question,
            "intent": None,
            "sentiment": None,
            "hits": [],
            "provider": self.provider.name,
            "tools_used": [],
            "agent_roles": [],
            "agent_trace": [],
            "answer": None,
            "status": "started",
            "mode": "linear",
        }

    def _classify(self, state: dict):

        question = state["question"]

        state["intent"] = self.intent_classifier.predict(question)
        state["sentiment"] = self.sentiment_classifier.predict(question)

        state["tools_used"].extend(["intent_classifier", "sentiment_classifier"])
        state["agent_roles"].append("coordinator")
        state["agent_trace"].append(f"classified intent as {state['intent']}")

        return state

    def _retrieve(self, state: dict):

        state["hits"] = self.retriever.search(state["question"], limit=3)

        state["tools_used"].append("vector_search")
        state["agent_roles"].append("retrieval_specialist")
        state["agent_trace"].append(f"retrieved {len(state['hits'])} knowledge hits")

        return state

    def _respond(self, state: dict):

        if state["sentiment"] and state["sentiment"].lower() == "negative":

            send_email_alert(state["question"])

            state["tools_used"].append("email_alert")
            state["agent_roles"].append("escalation_specialist")
            state["agent_trace"].append("sent escalation email")

        primary_answer = (
            state["hits"][0]["answer"]
            if state["hits"]
            else "I could not find an exact policy match in the knowledge base."
        )

        self.prompt.format_messages(
            intent=state["intent"],
            sentiment=state["sentiment"],
            question=state["question"],
            primary_answer=primary_answer,
        )

        try:

            answer, provider_trace = self.provider.generate(
                question=state["question"],
                context=state["hits"],
                intent=state["intent"],
                sentiment=state["sentiment"],
            )

            if provider_trace:
                state["agent_trace"].extend(provider_trace)

        except Exception as e:

            answer = f"AI provider error: {str(e)}"

        state["answer"] = answer

        state["agent_roles"].append("response_specialist")
        state["agent_trace"].append(f"generated final answer using {self.provider.name}")

        state["status"] = "completed"

        return state

    def _final(self, state: dict):

        return {
            "request_id": state["request_id"],
            "session_id": state["session_id"],
            "status": state["status"],
            "mode": state["mode"],
            "provider": state["provider"],
            "intent": state["intent"],
            "sentiment": state["sentiment"],
            "tools_used": state["tools_used"],
            "agent_roles": list(dict.fromkeys(state["agent_roles"])),
            "answer": state["answer"],
            "agent_trace": state["agent_trace"],
            "retrieval_hits": state["hits"],
        }

    def _run_linear(self, request_id: str, question: str, session_id: str | None):

        state = self._base_state(request_id, question, session_id)

        state = self._classify(state)
        state = self._retrieve(state)
        state = self._respond(state)

        return self._final(state)

    def _build_graph(self):

        from langgraph.graph import END, StateGraph

        graph = StateGraph(dict)

        graph.add_node("classify", self._classify)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("respond", self._respond)

        graph.set_entry_point("classify")

        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "respond")
        graph.add_edge("respond", END)

        return graph.compile()

    def _run_graph(self, request_id: str, question: str, session_id: str | None):

        if self._graph is None:
            self._graph = self._build_graph()

        state = self._base_state(request_id, question, session_id)
        state["mode"] = "graph"

        result = self._graph.invoke(state)

        return self._final(result)

    def _run_crew(self, request_id: str, question: str, session_id: str | None):

        state = self._base_state(request_id, question, session_id)
        state["mode"] = "crew"

        state = self._classify(state)
        state = self._retrieve(state)

        crew_answer = self.crew.run(
            question,
            state["hits"],
            state["intent"],
            state["sentiment"],
        )

        if crew_answer:

            state["answer"] = crew_answer
            state["agent_roles"].extend(["policy_specialist", "response_specialist"])
            state["agent_trace"].append("generated final answer using crewai")

            state["status"] = "completed"

            return self._final(state)

        state = self._respond(state)

        return self._final(state)