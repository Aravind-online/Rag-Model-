Agentic AI Assistant with RAG
A terminal-based AI assistant built with Python, Groq, FastEmbed, and Qdrant Cloud. It can answer general questions, perform arithmetic using a restricted calculator, retrieve information from local documents, and return the current time in a requested timezone.
Features
- LLM-powered assistant: Uses a Groq-hosted chat model configured through environment variables.
- Retrieval-Augmented Generation (RAG): Searches indexed document chunks and uses the retrieved text to generate grounded answers.
- Semantic search: Uses sentence-transformers/all-MiniLM-L6-v2 embeddings with Qdrant vector search.
- Supported files: .pdf, .txt, and .md.
- Calculator tool: Supports basic arithmetic, parentheses, floor division, modulo, and exponentiation without using Python eval().
- Timezone-aware clock: Retrieves the current time using IANA timezone names such as UTC and Asia/Kolkata.
- Conversation history: Keeps recent messages for follow-up questions during the current session.
- Configurable retrieval: Set the number of document chunks to retrieve using TOP_K.
Architecture
User
 |
 v
main.py  -- reads configuration and manages the chat loop
 |
 v
agent.py -- asks the Groq model whether to answer or call a tool
 |                         |
 |                         +--> Calculator
 |                         +--> Current-time tool
 |                         +--> Knowledge-base retrieval
 |                                      |
 v                                      v
Groq Chat API                      rag.py / FastEmbed
                                        |
                                        v
                                  Qdrant Cloud
                                        |
                                        v
                               Retrieved document chunks
                                        |
                                        v
                              Groq generates grounded answer
Project files
project/
├── main.py
├── agent.py
├── rag.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── data/
    ├── handbook.pdf
    ├── policies.md
    └── notes.txt
The data/ directory is resolved relative to the project layout in rag.py: it is expected to be one level above the directory containing rag.py. Adjust DATA_DIR if your project uses a different layout.
Tech Stack
Technology	Purpose
Python	Application logic
Groq API	Chat model and tool-call decisions
Qdrant Cloud	Vector storage and similarity search
FastEmbed	Text embeddings
all-MiniLM-L6-v2	Embedding model
python-dotenv	Load configuration from .env
pypdf	Extract text from PDF files


Requirements
- Python 3.10 or newer (Python 3.11 is a suitable choice)
- A Groq API key
- A Qdrant Cloud cluster and API key
- Internet access for API requests and Qdrant Cloud
- Enough local disk space for the embedding model and Python dependencies
Setup
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
Replace the placeholders with your repository URL and folder name.
2. Create and activate a virtual environment
Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, run this for the current terminal session and try again:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
If a requirements.txt file is included:
pip install -r requirements.txt
Otherwise, install the packages used by the project:
pip install groq python-dotenv qdrant-client fastembed pypdf
Use a compatible, tested version set for your environment and commit your project's requirements.txt so installations remain reproducible.
4. Configure environment variables
Create a file named .env in the directory from which you run main.py:
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

QDRANT_URL=https://your-qdrant-cluster-endpoint
QDRANT_API_KEY=your_qdrant_api_key

TOP_K=5
Get the Groq key from the Groq developer console and the Qdrant endpoint/API key from your Qdrant Cloud dashboard.
Notes:
- GROQ_MODEL is optional; the code defaults to openai/gpt-oss-20b.
- TOP_K is optional; the code defaults to 5 and must be a positive integer.
- QDRANT_URL must use HTTPS. The current code appends port 6333 if the URL has no explicit port. Confirm that this matches the endpoint and network configuration for your Qdrant deployment.
- Never commit real API keys to source control.
5. Add documents
Place your documents in the data/ directory at the path expected by rag.py.
Supported formats:
- PDF (.pdf)
- Plain text (.txt)
- Markdown (.md)
On startup, the application discovers supported files, extracts their text, splits the text into chunks, generates embeddings, and stores the chunks in Qdrant.
PDF note: Text extraction works for PDFs containing selectable text. Scanned/image-only PDFs may require OCR before indexing.
6. Run the assistant
From the directory containing main.py:
python main.py
On startup, the program validates configuration, initializes Qdrant, indexes the documents it finds, and opens the interactive terminal.
Usage
Example interaction:
============================================================
🤖 Agentic AI
============================================================
Model: openai/gpt-oss-20b
Available tools:
  🧮 Calculator
  🕒 Current date and time
  📚 Search indexed documents

Type 'exit' or 'quit' to stop.
Type 'clear' to clear the conversation.
============================================================

👤 You: Calculate (250 * 18) / 5

🤖 Agent: 900

👤 You: What does the indexed handbook say about attendance?

🤖 Agent: [Answer based on the retrieved document context]

👤 You: What time is it in Asia/Kolkata?

🤖 Agent: [Current time in the requested timezone]
The answers above are illustrative; actual responses depend on the configured model, available documents, and tool results.
Built-in commands
Input	Action
exit or quit	Stop the assistant
clear	Clear the in-memory conversation history


Calculator support
The calculator supports:
+   Addition
-   Subtraction
*   Multiplication
/   Division
//  Floor division
%   Modulo
**  Exponentiation
()  Parentheses
It rejects unsupported Python syntax and applies basic expression-length, syntax-tree complexity, and exponent limits. These safeguards reduce risk but should not be treated as a complete sandbox for untrusted inputs.
Timezones
The time tool uses IANA timezone identifiers. Examples:
UTC
Asia/Kolkata
Asia/Tokyo
Europe/London
America/New_York
The system prompt instructs the model to use UTC unless the user specifies another location or timezone.
How RAG works
1. Load: Read PDF, Markdown, and text files from the configured data directory.
2. Chunk: Normalize whitespace and split document text into overlapping chunks.
3. Embed: Convert each chunk into a vector using all-MiniLM-L6-v2 through FastEmbed.
4. Index: Store vectors and payloads (source and document) in the Qdrant agent_documents collection.
5. Retrieve: Embed the user's search query and request the closest vectors from Qdrant.
6. Build context: Format retrieved chunks with their source paths.
7. Generate: Send the question and retrieved context to the Groq model to produce an answer.
Current chunk settings in rag.py:
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
These values are character-based, not token-based. You may want to evaluate larger chunks and different overlap sizes for longer documents.
Important implementation notes
- run_agent() gives the model access to the three declared tools using tool_choice="auto".
- The calculator and time tools return their results to the model for a follow-up response.
- The knowledge-base path retrieves chunks and calls generate_answer() directly; it does not return to the general tool loop after retrieval.
- The application retains conversation history in memory only. It is cleared when the user enters clear and is not automatically persisted between runs.
- Re-indexing a source deletes that source's previously indexed points before upserting the new chunks. Deleted or renamed local files are not automatically reconciled with Qdrant by the current implementation.
- Retrieval currently does not enforce a minimum similarity score, so the closest returned chunks may not always be relevant.
- The system prompt encourages answers grounded in retrieved text, but prompt instructions alone cannot guarantee factual correctness.
Troubleshooting
GROQ_API_KEY not found
- Confirm .env is in the directory where load_dotenv() can find it.
- Check that the variable is spelled GROQ_API_KEY.
- Restart the terminal after changing environment configuration.
Qdrant connection errors
- Verify QDRANT_URL and QDRANT_API_KEY.
- Ensure the URL uses HTTPS and matches the cluster endpoint.
- Check whether your network can reach the configured endpoint and port.
- Confirm the API key has permission to access the collection.
No documents found
- Check the expected data/ directory.
- Verify that files end in .pdf, .md, or .txt.
- Confirm the files contain extractable text.
PDF returns no text
The PDF may be scanned or image-based. Run OCR first or use a PDF with selectable text.
Missing Python modules
Activate the correct virtual environment and install dependencies:
pip install -r requirements.txt
Model or API errors
- Check the Groq model name and API key.
- Verify API availability, rate limits, and network connectivity.
- Review the error printed in the terminal.
Security
- Keep .env, API keys, and other secrets out of Git.
- Do not commit private or confidential documents to a public repository.
- Treat retrieved document content as untrusted input.
- Validate tool arguments and apply appropriate request, timeout, and resource limits before exposing the application to other users.
Suggested .gitignore entries:
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
Future Improvements
- Add a web UI with Streamlit or a frontend framework.
- Add document upload and re-indexing controls.
- Reconcile deleted and renamed documents in Qdrant.
- Add retrieval-score thresholds and evaluate search quality.
- Improve chunking with sentence-aware or token-aware splitting.
- Validate tool arguments against strict schemas.
- Add automated tests, structured logging, and citations in answers.
- Support OCR for scanned PDFs.
- Add conversation persistence if needed.
License
Choose and add a license before distributing this project publicly. If no license is included, others do not automatically receive permission to reuse, modify, or redistribute the code.
