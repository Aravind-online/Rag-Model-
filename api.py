import os
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq, GroqError
from pydantic import BaseModel, Field

from fastapi.staticfiles import StaticFiles

from agent import run_agent
from rag import KnowledgeBase


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    history: list[ChatMessage] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required. Add it to your .env file.")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required in your .env file.")

    try:
        top_k = int(os.getenv("TOP_K", "5"))
        if top_k < 1:
            raise ValueError
    except ValueError as error:
        raise RuntimeError("TOP_K must be a positive integer.") from error

    knowledge_base = KnowledgeBase(qdrant_url, qdrant_api_key)
    try:
        indexed_chunks = knowledge_base.index_documents()
        if indexed_chunks:
            print(f"Indexed {indexed_chunks} document chunk(s) into Qdrant.")
    except Exception as e:
        print(f"WARNING: Could not connect to Qdrant Cloud on startup: {type(e).__name__}")
        print("   The server will start, but RAG search will be unavailable until Qdrant is reachable.")
        print("   Check that your Qdrant cluster is active at https://cloud.qdrant.io")

    app.state.client = Groq(api_key=api_key, timeout=30.0)
    app.state.model = model
    app.state.knowledge_base = knowledge_base
    app.state.top_k = top_k
    yield


app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest, request: Request) -> dict[str, str]:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    messages = [
        {"role": item.role, "content": item.content}
        for item in payload.history[-20:]
    ]
    messages.append({"role": "user", "content": message})

    try:
        answer = run_agent(
            request.app.state.client,
            request.app.state.model,
            messages,
            knowledge_base=request.app.state.knowledge_base,
            top_k=request.app.state.top_k,
        )
    except GroqError as error:
        raise HTTPException(status_code=502, detail="The language model request failed.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="The assistant could not process this request.") from error

    return {"answer": answer}


frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")