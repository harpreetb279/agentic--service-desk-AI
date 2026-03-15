from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from mangum import Mangum
from pydantic import BaseModel, Field

from models.agent_pipeline import SupportAgent

BASE_DIR = Path(__file__).resolve().parent
START_TS = time.time()

try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, generate_latest
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OTEL_AVAILABLE = True
except Exception:
    OTEL_AVAILABLE = False

if PROM_AVAILABLE:
    REQUEST_COUNT = Counter("servicedesk_requests_total", "Total ServiceDesk API requests")
    REQUEST_LATENCY = Histogram("servicedesk_latency_seconds", "ServiceDesk API latency")


def _split_origins(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


allowed_origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "https://harpreetbrar.dev",
    "https://www.harpreetbrar.dev",
    "https://servicedesk.harpreetbrar.dev",
    "https://resume.harpreetbrar.dev",
    "https://gpt.harpreetbrar.dev",
    "https://soc.harpreetbrar.dev",
    "https://detectfraud.harpreetbrar.dev",
    "https://mediscribe.harpreetbrar.dev",
]

allowed_origins = list(
    dict.fromkeys(
        allowed_origins + _split_origins(os.getenv("CORS_ALLOWED_ORIGINS", ""))
    )
)

app = FastAPI(title="ServiceDesk AI", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if OTEL_AVAILABLE:
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )
    FastAPIInstrumentor.instrument_app(app)

_agent: SupportAgent | None = None


def get_agent() -> SupportAgent:
    global _agent
    if _agent is None:
        _agent = SupportAgent(base_dir=BASE_DIR)
    return _agent


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str | None = Field(default="auto")
    session_id: str | None = Field(default=None)


class AlertRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def run_crewai_orchestration(question: str):
    if not CREWAI_AVAILABLE:
        return {"crewai": "not_installed"}

    try:
        from crewai import LLM

        gemini_llm = LLM(
            model=f"gemini/{os.getenv('CREWAI_GEMINI_MODEL', 'gemini-3')}",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        )

        coordinator = CrewAgent(
            role="Support Request Coordinator",
            goal="Understand user request and classify intent",
            backstory="Senior helpdesk coordinator",
            llm=gemini_llm,
            verbose=False,
        )

        retrieval_specialist = CrewAgent(
            role="Knowledge Retrieval Specialist",
            goal="Search internal knowledge base",
            backstory="Documentation expert",
            llm=gemini_llm,
            verbose=False,
        )

        response_specialist = CrewAgent(
            role="Customer Response Specialist",
            goal="Generate helpful user response",
            backstory="Customer support expert",
            llm=gemini_llm,
            verbose=False,
        )

        classify_task = CrewTask(
            description=f"Classify intent of support request: {question}",
            expected_output="A short intent classification.",
            agent=coordinator,
        )

        retrieve_task = CrewTask(
            description=f"Retrieve relevant knowledge for: {question}",
            expected_output="A short summary of relevant knowledge.",
            agent=retrieval_specialist,
        )

        respond_task = CrewTask(
            description=f"Generate final response for: {question}",
            expected_output="A concise user-facing support answer.",
            agent=response_specialist,
        )

        crew = Crew(
            agents=[coordinator, retrieval_specialist, response_specialist],
            tasks=[classify_task, retrieve_task, respond_task],
            verbose=False,
        )

        crew.kickoff()

        return {
            "crewai": "executed",
            "llm": "gemini",
            "agents": [
                "coordinator",
                "retrieval_specialist",
                "response_specialist",
            ],
        }

    except Exception as exc:
        return {"crewai_error": str(exc)}


@app.get("/")
async def home():
    return {
        "message": "ServiceDesk AI API is running",
        "service": "servicedesk-ai",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "servicedesk-ai",
        "uptime_s": int(time.time() - START_TS),
        "agent_loaded": _agent is not None,
    }


@app.get("/ready")
async def ready():
    try:
        agent = get_agent()
        return {
            "ready": True,
            "details": agent.readiness(),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "error": str(exc)},
        )


@app.get("/faqs")
async def faqs():
    agent = get_agent()
    return agent.get_faq_questions()


@app.post("/ask")
async def ask(payload: AskRequest):
    started = time.time()

    try:
        agent = get_agent()
        selected_mode = (payload.mode or "auto").lower()
        crew_trace = None

        if selected_mode == "crew" and agent.crew_enabled:
            crew_trace = run_crewai_orchestration(payload.question)

        result = agent.handle_query(
            payload.question,
            mode=selected_mode,
            session_id=payload.session_id,
        )

        if PROM_AVAILABLE:
            REQUEST_COUNT.inc()
            REQUEST_LATENCY.observe(time.time() - started)

        if isinstance(result, dict):
            result["crew_trace"] = crew_trace
            return result

        return {
            "result": result,
            "crew_trace": crew_trace,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/alerts")
async def alerts(payload: AlertRequest):
    from notify import send_email_alert

    result = send_email_alert(payload.query)
    return {
        "status": "ok",
        "result": result,
    }


@app.get("/debug")
async def debug():
    agent = get_agent()
    return agent.debug_state()


@app.get("/metrics")
async def metrics():
    if not PROM_AVAILABLE:
        return {"metrics": "prometheus_not_installed"}

    return Response(generate_latest(), media_type="text/plain")


@app.get("/__version")
async def version():
    agent = get_agent()
    return {
        "service": "servicedesk-ai",
        "version": app.version,
        "provider": agent.provider.name,
        "vector_backend": agent.retriever.backend_name,
        "langchain_available": agent.feature_flags["langchain_installed"],
        "langgraph_available": agent.feature_flags["langgraph_installed"],
        "crewai_available": agent.feature_flags["crewai_installed"],
        "langfuse_available": agent.feature_flags["langfuse_installed"],
        "openai_available": agent.feature_flags["openai_installed"],
        "gemini_available": agent.feature_flags["gemini_installed"],
    }


handler = Mangum(app, api_gateway_base_path="/prod")