import streamlit as st
from rag_pipeline import (
    load_pdfs,
    chunk_documents,
    get_embedding_model,
    build_vectorstore,
    answer_question,
)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Q&A Chatbot")
st.caption("Upload PDFs, then ask questions answered strictly from their content.")

# --- Sidebar: API key + document upload ---
with st.sidebar:
    st.header("Setup")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at console.groq.com",
    )
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )
    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "This app embeds your PDFs locally with Sentence Transformers, "
        "indexes them in FAISS for semantic search, and generates answers "
        "using Llama 3.1 via the Groq API."
    )

# --- Session state ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Process documents ---
if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    else:
        with st.spinner("Reading and indexing documents..."):
            documents = load_pdfs(uploaded_files)
            chunks = chunk_documents(documents)
            embedding_model = get_embedding_model()
            st.session_state.vectorstore = build_vectorstore(chunks, embedding_model)
        st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**, page {src['page']}")
                    st.caption(src["snippet"])

# --- Chat input ---
query = st.chat_input("Ask a question about your documents...")

if query:
    if st.session_state.vectorstore is None:
        st.error("Please upload and process at least one PDF before asking a question.")
    elif not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(query, st.session_state.vectorstore, groq_api_key)
                st.markdown(answer)

                source_info = []
                with st.expander("Sources"):
                    for doc in sources:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:200] + "..."
                        st.markdown(f"**{source}**, page {page}")
                        st.caption(snippet)
                        source_info.append({"source": source, "page": page, "snippet": snippet})

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": source_info}
        )
