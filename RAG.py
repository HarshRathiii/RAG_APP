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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


st.set_page_config(page_title="PDF & Webpage Q&A", layout="centered")
st.title("📄🔗 Multi PDF & Webpage Vectorizer")

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

st.write(f"🔹 Total documents loaded: {len(all_docs)}")

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

st.write(f"🔹 Total chunks 5: {len(all_chunks)}")

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

st.success("✅ All chunks converted into embeddings and stored in Pinecone")



import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import AIMessage, HumanMessage
from gtts import gTTS

from langchain.chat_models import init_chat_model
llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq")

retriever = vectorstore.as_retriever()

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grounded RAG assistant. Answer strictly using the provided context from the uploaded PDFs or websites. Always cite the source IDs like [S#] when giving information. 

Rules:
1. Only use the retrieved context to answer. Do not invent or assume facts. 
2. If the answer is not fully supported by the context, say: "I don’t have enough information in the uploaded documents to answer that." Then offer to search more uploaded files or run a generic web search only if allowed. 
3. Never contradict the sources. If they disagree, summarise both sides and cite them. 
4. Stay strictly on the topic of the uploaded content. If the user asks something unrelated, refuse and say it’s out of scope, unless generic search is explicitly enabled. 
5. Be concise, accurate, and neutral. If listing steps or details, use short bullet points. 

If context is missing or insufficient then provide profound information"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# ------------------------
# 4. Prompt to answer questions
# ------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use the retrieved context to answer the user's question.5"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("system", "Context:\n{context}")
])

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

# 5. LangGraph wrapper to manage history
# ------------------------
def call_rag_chain(state: MessagesState):
    user_input = state["messages"][-1].content
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": state["messages"]
    })
    return {"messages": [("ai", response["answer"])]}

# Define the graph
workflow = StateGraph(MessagesState)
workflow.add_node("rag", call_rag_chain)
workflow.add_edge(START, "rag")
workflow.add_edge("rag", END)

# Add memory (history storage)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat1"  # unique per user/session

user_input = st.chat_input("Ask a question...")
if user_input:
    events = app.invoke(
        {"messages": [("user", user_input)]},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
    )


    for event in events["messages"]:
        if isinstance(event, AIMessage):
            st.chat_message("assistant").write(event.content)

            # --- NEW: Convert to speech ---
            tts = gTTS(event.content, lang="en")
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")

        elif isinstance(event, HumanMessage):
            st.chat_message("user").write(event.content)










