import streamlit as st
import os

os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


st.set_page_config(page_title="PDF & Webpage Q&A", layout="centered")
st.title("TUTOR")

# Step 1: Upload multiple PDFs

pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Step 2: Enter multiple URLs

urls_text = st.text_area("Enter webpage URLs (one per line)")
urls = [url.strip() for url in urls_text.split("\n") if url.strip()]

all_docs = []

# Load PDFs
for pdf_file in pdf_files:
    with open(f"temp_{pdf_file.name}", "wb") as f:
        f.write(pdf_file.getvalue())
    loader = PyPDFLoader(f"temp_{pdf_file.name}")
    docs = loader.load()
    all_docs.extend(docs)

# Load webpages
for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    all_docs.extend(docs)

st.write(f" Total documents loaded: {len(all_docs)}")

# Step 3: Split into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
all_chunks = []
for doc in all_docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

st.write(f"Total chunks 5: {len(all_chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"}) 
# Create or connect to an index
index_name = "rag-app-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # embedding size for MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=st.secrets["PINECONE_ENVIRONMENT"])
    )

# Use the index
index = pc.Index(index_name)

# Store your embeddings in Pinecone
vectorstore = PineconeVectorStore.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    index_name=index_name
)

st.success("All chunks converted into embeddings and stored in Pinecone")



import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from gtts import gTTS

from langchain_groq import ChatGroq

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
    ("system", """You are an assistant that rewrites follow-up questions into fully self-contained questions.

Your job:
- Replace pronouns like "he", "she", "it", "they", "his", "that", etc. with the correct subject from chat history.
- Make the question complete and clear.
- Preserve the original meaning.
- Do NOT answer the question.

Example:
Chat History: "Virat Kohli is a cricketer."
Follow-up: "What is his age?"
Output: "What is Virat Kohli's age?"
"""),
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
