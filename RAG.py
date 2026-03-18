import streamlit as st
import os

# ------------------------
# ENV VARIABLES
# ------------------------
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"

# ------------------------
# IMPORTS
# ------------------------
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from gtts import gTTS

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="RAG Tutor", layout="centered")
st.title("📄 RAG Tutor (PDF + Web)")

# ------------------------
# INPUTS
# ------------------------
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

urls_text = st.text_area("Enter webpage URLs (one per line)")
urls = [url.strip() for url in urls_text.split("\n") if url.strip()]

# ------------------------
# LOAD DOCUMENTS
# ------------------------
all_docs = []

# PDFs
for pdf_file in pdf_files:
    with open(f"temp_{pdf_file.name}", "wb") as f:
        f.write(pdf_file.getvalue())
    loader = PyPDFLoader(f"temp_{pdf_file.name}")
    docs = loader.load()
    all_docs.extend(docs)

# URLs
for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    all_docs.extend(docs)

st.write(f"📄 Documents loaded: {len(all_docs)}")

# ------------------------
# SPLIT
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []
for doc in all_docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

st.write(f"🧩 Total chunks: {len(all_chunks)}")

# ------------------------
# EMBEDDINGS + PINECONE
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "rag-app-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=st.secrets["PINECONE_ENVIRONMENT"]
        )
    )

vectorstore = PineconeVectorStore.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    index_name=index_name
)

st.success("✅ Embeddings stored in Pinecone")

# ------------------------
# LLM
# ------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
)

# ------------------------
# PROMPTS
# ------------------------

# Query rewriting
CONDENSE_PROMPT = PromptTemplate.from_template("""
Given the chat history and a follow-up question, rewrite it into a standalone question.

Chat History:
{chat_history}

Follow-up question:
{question}

Standalone question:
""")

# Strict RAG QA
QA_PROMPT = PromptTemplate.from_template("""
You are a RAG assistant.

Rules:
- Use ONLY the provided context
- Do NOT use outside knowledge
- If answer not found, say:
"I don’t have enough information in the provided documents to answer that."

Context:
{context}

Question:
{question}

Answer:
""")

# ------------------------
# RAG CHAIN
# ------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    condense_question_prompt=CONDENSE_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False
)

# ------------------------
# CHAT HISTORY
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ------------------------
# CHAT INPUT
# ------------------------
user_input = st.chat_input("Ask a question...")

if user_input:
    result = qa_chain({
        "question": user_input,
        "chat_history": st.session_state["chat_history"]
    })

    answer = result["answer"]

    # Save history
    st.session_state["chat_history"].append((user_input, answer))

# ------------------------
# DISPLAY CHAT
# ------------------------
for q, a in st.session_state["chat_history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

    if a.strip():
        tts = gTTS(a, lang="en")
        tts.save("output.mp3")
        st.audio("output.mp3")
