# RAG Document Q&A Chatbot

Upload PDF documents and ask questions answered strictly from their content, using
semantic search (not just keyword matching) to find the relevant passages.

## How it works

1. **Load & chunk** — PDFs are parsed and split into overlapping text chunks so
   context isn't severed at arbitrary boundaries.
2. **Embed** — Each chunk is converted into a vector embedding using a
   Sentence Transformer model (runs locally, no API needed).
3. **Index** — Embeddings are stored in a FAISS vector index for fast similarity search.
4. **Retrieve** — When you ask a question, it's embedded the same way, and the
   top-k most similar chunks are pulled from the index.
5. **Generate** — Those chunks are passed as context to a Llama 3.1 model hosted on
   Groq, which generates an answer grounded in the retrieved text.

## Tech stack

Python · Streamlit · LangChain · FAISS · Sentence Transformers · Groq (Llama 3.1)

## Run locally

```bash
git clone <your-repo-url>
cd rag-chatbot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, paste in your Groq API key (get one free at
[console.groq.com](https://console.groq.com)), upload a PDF, click **Process
Documents**, then start asking questions.

## Deploy for free (Streamlit Community Cloud)

1. Push this project to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file
   path to `app.py`.
4. Click **Deploy**. You'll get a live public URL like
   `https://your-app-name.streamlit.app`.
5. Your Groq API key is entered per-session in the app's sidebar — it is never
   stored in the repo, so there's nothing extra to configure as a "secret" for
   this basic setup. (If you want to hardcode a default key instead, add it
   under the app's **Settings → Secrets** in the Streamlit Cloud dashboard,
   not in the code itself.)

## Notes

- The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first
  run — this requires an internet connection to Hugging Face the first time
  the app starts.
- Groq's free tier has generous rate limits, which is what makes this
  deployable without cost.
