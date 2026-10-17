import os

from dotenv import load_dotenv
from groq import Groq, GroqError
from qdrant_client.http.exceptions import ResponseHandlingException

from agent import run_agent
from rag import KnowledgeBase


# Load environment variables
load_dotenv()

# Get configuration
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
try:
    top_k = int(os.getenv("TOP_K", "5"))
    if top_k < 1:
        raise ValueError
except ValueError:
    print("❌ TOP_K must be a positive integer.")
    exit(1)

if not api_key:
    print("❌ GROQ_API_KEY not found.")
    print("Add it to your .env file:")
    print("GROQ_API_KEY=your_api_key_here")
    exit(1)

if not qdrant_url or not qdrant_api_key:
    print("❌ QDRANT_URL and QDRANT_API_KEY are required for Qdrant Cloud.")
    print("Add both values from your Qdrant Cloud cluster to your .env file.")
    exit(1)

# Create Groq client
client = Groq(api_key=api_key, timeout=30.0)
try:
    knowledge_base = KnowledgeBase(qdrant_url, qdrant_api_key)
    indexed_chunks = knowledge_base.index_documents()
except ResponseHandlingException as error:
    print("❌ Could not connect to Qdrant Cloud.")
    print("Check that QDRANT_URL is the active cluster endpoint and that HTTPS port 6333 is reachable.")
    print("Connection error:", type(error.source).__name__)
    exit(1)
except Exception as error:
    print("❌ Could not initialize the Qdrant knowledge base.")
    print("Check QDRANT_URL, QDRANT_API_KEY, and document indexing configuration.")
    print("Error:", type(error).__name__)
    exit(1)
if indexed_chunks:
    print(f"Indexed {indexed_chunks} document chunk(s) into Qdrant.")

# Conversation history
messages = []


def main():
    print("=" * 60)
    print("🤖 Agentic AI")
    print("=" * 60)
    print("Model:", model)
    print("Available tools:")
    print("  🧮 Calculator")
    print("  🕒 Current date and time")
    print("  📚 Search indexed documents")
    print()
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'clear' to clear the conversation.")
    print("=" * 60)

    while True:

        try:
            prompt = input("\n👤 You: ").strip()

            if not prompt:
                continue

            # Exit
            if prompt.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break

            # Clear conversation
            if prompt.lower() == "clear":
                messages.clear()
                print("🧹 Conversation cleared.")
                continue

            # Add user message
            messages.append({
                "role": "user",
                "content": prompt
            })

            print("\n🤖 Agent: Thinking...")

            # Run agent
            answer = run_agent(
                client,
                model,
                messages,
                knowledge_base=knowledge_base,
                top_k=top_k,
            )

            print("\n🤖 Agent:", answer)

            # Add assistant response
            messages.append({
                "role": "assistant",
                "content": answer
            })

        except GroqError as error:
            print("\n❌ Groq request failed:")
            print(error)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

        except Exception as error:
            print("\n❌ Error:")
            print(error)


if __name__ == "__main__":
    main()