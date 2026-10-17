from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from urllib.parse import urlsplit, urlunsplit

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COLLECTION_NAME = "agent_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_SUFFIXES = {".md", ".pdf", ".txt"}
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
RAG_SYSTEM_PROMPT = """You answer questions using retrieved knowledge-base context.
Use the provided context as the factual source. Do not invent information. If the
answer is not available in the retrieved context, clearly say it is not available
in the knowledge base. Prefer concise, direct answers, combine relevant details
from multiple chunks, preserve important facts, and do not copy large passages.
Conversation history may clarify a follow-up question, but must not be used to
introduce unsupported facts. If context sources conflict, explicitly describe
the conflict. Do not expose retrieval, embeddings, vector IDs, API keys, or
implementation details, and do not mention Qdrant in a normal answer."""
NOT_AVAILABLE_ANSWER = "That information is not available in the knowledge base."


def build_context(results: list[dict[str, str]]) -> str:
    """Format retrieved chunks with explicit source boundaries."""
    return "\n\n".join(
        f"[Source {index}: {result['source']}]\n{result['document']}"
        for index, result in enumerate(results, start=1)
        if result.get("document", "").strip()
    )


def generate_answer(
    client,
    model: str,
    query: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Generate a concise answer grounded in retrieved context."""
    if not context.strip():
        return NOT_AVAILABLE_ANSWER

    conversation = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    if history:
        conversation.extend(
            message
            for message in history[-10:]
            if message.get("role") in {"user", "assistant"}
        )
    conversation.append(
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nRetrieved context:\n{context}",
        }
    )
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer.strip() if answer else "I couldn't generate an answer from the available context."


class KnowledgeBase:
    def __init__(self, url: str, api_key: str) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        parsed_url = urlsplit(url.rstrip("/"))
        if parsed_url.scheme != "https":
            raise ValueError("QDRANT_URL must use HTTPS for Qdrant Cloud.")
        if parsed_url.port is None:
            url = urlunsplit(
                (parsed_url.scheme, f"{parsed_url.netloc}:6333", parsed_url.path, parsed_url.query, parsed_url.fragment)
            )
        self.client = QdrantClient(url=url, api_key=api_key)
        self._embedding_model: TextEmbedding | None = None

    @property
    def embedding_model(self) -> TextEmbedding:
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        return self._embedding_model

    def index_documents(self) -> int:
        files = sorted(
            path
            for path in DATA_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not files:
            print("No documents found. Add .txt, .md, or .pdf files to the data folder.")
            return 0

        indexed_chunks = 0
        for path in files:
            try:
                text = self._read_document(path)
            except Exception as error:
                print(f"Skipping {path.name}: {error}")
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                continue

            embeddings = [embedding.tolist() for embedding in self.embedding_model.embed(chunks)]
            self._ensure_collection(len(embeddings[0]))
            source = path.relative_to(DATA_DIR).as_posix()
            if self.client.collection_exists(COLLECTION_NAME):
                self.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source", match=models.MatchValue(value=source)
                            )
                        ]
                    ),
                )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, f"{source}:{chunk_index}")),
                        vector=embedding,
                        payload={"source": source, "document": chunk},
                    )
                    for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ],
            )
            indexed_chunks += len(chunks)

        return indexed_chunks

    def retrieve_documents(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        if not self.client.collection_exists(COLLECTION_NAME):
            return []

        query_embedding = next(self.embedding_model.query_embed(query)).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
        ).points
        documents = []
        for result in results:
            payload = result.payload or {}
            document = payload.get("document", "")
            if document.strip():
                documents.append(
                    {
                        "source": str(payload.get("source", "unknown")),
                        "document": document,
                    }
                )
        return documents

    def search(self, query: str, limit: int = 5) -> str:
        """Return formatted search results for compatibility with existing callers."""
        documents = self.retrieve_documents(query, limit)
        if not documents:
            return NOT_AVAILABLE_ANSWER
        return build_context(documents)

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        payload_schema = self.client.get_collection(COLLECTION_NAME).payload_schema
        if "source" not in payload_schema:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    @staticmethod
    def _read_document(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        text = " ".join(text.split())
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
        return chunks