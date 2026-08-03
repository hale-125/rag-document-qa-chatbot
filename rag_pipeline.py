"""
Core RAG pipeline: document loading, chunking, embedding, vector storage,
retrieval, and answer generation.
"""
import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
GROQ_MODEL = "llama-3.1-8b-instant"


def load_pdfs(uploaded_files) -> List[Document]:
    """Load one or more uploaded PDF files into LangChain Documents,
    tagging each with its source filename for citation."""
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks so context isn't severed
    at arbitrary boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embedding_model():
    """Load the sentence-transformer embedding model (downloaded once,
    cached locally by HuggingFace on first run)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(chunks: List[Document], embedding_model) -> FAISS:
    """Embed chunks and index them in FAISS for fast similarity search."""
    return FAISS.from_documents(chunks, embedding_model)


def get_llm(groq_api_key: str):
    """Return a Groq-hosted LLM client. Groq runs the model on its own
    servers, so no local GPU/CPU inference is needed -- this is what
    makes the app deployable on free hosting."""
    return ChatGroq(
        api_key=groq_api_key,
        model=GROQ_MODEL,
        temperature=0.2,
    )


PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below, \
which was retrieved from the user's uploaded documents. If the answer isn't in the context, \
say you don't have enough information rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(query: str, vectorstore: FAISS, groq_api_key: str, k: int = 4):
    """Retrieve the top-k most relevant chunks for the query, then ask
    the LLM to answer using only that retrieved context. Returns the
    answer text plus the source chunks used, for citation in the UI."""
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    llm = get_llm(groq_api_key)
    response = llm.invoke(prompt)

    return response.content, retrieved_docs
