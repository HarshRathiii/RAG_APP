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
    api_key=os.environ["GROQ_API_KEY"],
)

retriever = vectorstore.as_retriever()

# ------------------------
# 1. QUERY REWRITING PROMPT (GENERAL)
# ------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """Rewrite the user's latest question into a clear standalone question 
based on the chat history. Do NOT answer the question."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# ------------------------
# 2. QA PROMPT (STRICT RAG)
# ------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do NOT use outside knowledge.
- If the answer is not in the context, say:
  "I don’t have enough information in the provided documents to answer that."
- Keep answers clear and concise.

Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

# ------------------------
# CHAT HISTORY (SAFE FORMAT)
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ------------------------
# USER INPUT
# ------------------------
user_input = st.chat_input("Ask a question...")

if user_input:
    # Display user message
    st.chat_message("user").write(user_input)

    # Convert history to LangChain format
    history_for_chain = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user"
        else AIMessage(content=msg["content"])
        for msg in st.session_state["messages"]
    ]

    # Invoke RAG chain
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": history_for_chain
    })

    answer = response.get("answer", "").strip()

    # Store messages as DICTS (prevents JSON error)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Display assistant response
    if answer:
        st.chat_message("assistant").write(answer)

        # TTS safe
        tts = gTTS(answer, lang="en")
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")
