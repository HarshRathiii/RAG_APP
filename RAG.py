import os
import streamlit as st
from gtts import gTTS
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------------
# Streamlit page setup
# ------------------------
st.set_page_config(page_title="RAG PDF/Web Q&A", layout="centered")
st.title("RAG Application")

# ------------------------
# Environment setup
# ------------------------
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"

# Pinecone setup
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# ------------------------
# Step 1: Upload PDFs
# ------------------------
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
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

st.write(f"Total documents loaded: {len(all_docs)}")

# ------------------------
# Step 2: Split into chunks
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
all_chunks = []
for doc in all_docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

st.write(f"Total chunks: {len(all_chunks)}")

# ------------------------
# Step 3: Create embeddings & Pinecone index
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

index_name = "rag-app-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=st.secrets["PINECONE_ENVIRONMENT"])
    )

vectorstore = PineconeVectorStore.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    index_name=index_name
)

st.success("Embeddings stored in Pinecone!")

# ------------------------
# Step 4: Setup Groq LLM + retrieval
# ------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
)

retriever = vectorstore.as_retriever()

# Prompt to rewrite user query into standalone query
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a RAG system.
Rewrite the user's latest question into a self-contained query for document retrieval.
Do NOT answer the question.
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# Prompt to answer user question using retrieved documents
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for a RAG system.
Answer the user's question using ONLY the provided context.
If context does not contain the answer, say: 
"I don’t have enough information in the provided documents to answer that."
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
# Step 5: Multi-turn chat
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.chat_input("Ask a question...")

if user_input:
    # Save user message as plain dict
    user_msg = {"role": "user", "content": user_input}
    st.session_state["chat_history"].append(user_msg)

    # Call RAG chain
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state["chat_history"]
    })

    answer = response.get("answer", "").strip()
    if answer:
        # Save assistant message as plain dict
        assistant_msg = {"role": "assistant", "content": answer}
        st.session_state["chat_history"].append(assistant_msg)

        # Display in chat
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(answer)

        # TTS
        tts = gTTS(answer, lang="en")
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")
