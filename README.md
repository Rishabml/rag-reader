# RAG Reader 📚

Chat with your PDF documents while verifying answers directly from the source page.
Upload any document, ask questions in natural language, and get answers grounded in your content — with the exact page reference to verify.

## Live Demo
👉 [rag-reader-chatbot.streamlit.app](https://rag-reader-chatbot.streamlit.app)

## What it does
- Upload a PDF document
- Ask questions in natural language
- Get answers powered by LLaMA 4 Scout via Groq
- See the exact source page the answer came from
- Maintains conversational context across questions

## Tech Stack
- **LLM** — Groq `meta-llama/llama-4-scout-17b-16e-instruct`
- **Orchestration** — LangChain + langchain-classic
- **Vector Store** — ChromaDB
- **Embeddings** — HuggingFace `sentence-transformers/all-mpnet-base-v2`
- **Frontend** — Streamlit

## What I learned building this

**1. Streamlit's execution model** — Every user interaction triggers a full top-to-bottom script rerun. Understanding this changed how I thought about where to put logic vs where to put rendering. Heavy operations need to be gated behind buttons or cached — not left running freely in the script.

**2. State management vs caching** — `st.session_state` and `@st.cache_resource` solve different problems. Session state is per user and per session — right for vector stores and chat history where user data must stay isolated. Cache resource is shared across all users — right for the HuggingFace model which is identical for everyone.

**3. Conversational chains in LangChain** — How `ConversationalRetrievalChain` manages chat history and passes context between turns, and why separating the retriever from the LLM gives you control over both independently.

## Known Limitations
- Single PDF per session — uploading a new PDF requires reprocessing
- No persistent chat history across sessions
- Vector store is rebuilt in memory on every upload — not persisted to disk
- HuggingFace embeddings model loads on every new session
- App has no rate limiting — API tokens are consumed on every query

## Run Locally

```bash
git clone https://github.com/Rishabml/rag-reader.git
cd rag-reader
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

```bash
streamlit run app.py
```

## Connect
[LinkedIn](https://www.linkedin.com/in/rishabh-gangwar-448134394)