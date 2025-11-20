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
    ("system", """You are a RAG-based educational assistant designed to help autistic learners.
Your primary goal is to explain and generate information in a way that is:
Clear, structured, and easy to follow.
Free from unnecessary jargon or ambiguity.
Visually or step-wise presented wherever possible.
Emotionally neutral and supportive.
You must use only the retrieved context from the uploaded PDFs or websites when answering. Always cite the source IDs like [S#].

Rules:
1. Use only the provided context to answer. Do not assume or invent facts.
2. If the answer is missing or incomplete, say: "I don’t have enough information in the uploaded documents to answer that."
3. Then offer to explore more uploaded files or suggest a web search (only if allowed).
4. If sources disagree, summarize both sides clearly and cite them.
5. Stay strictly within the context topic; if unrelated, politely state that it’s out of scope.
6. Keep tone simple, kind, and autism-friendly — short sentences, bullet points, and clear formatting.
7. Provide explanations and examples that make concepts easier to understand.
8. If context is missing, give well-structured, easy-to-understand general information for learning."""),
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
    ("system", "You are an AI assistant. Use the retrieved context to answer the user's question accurately and concisely."),
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

from langchain_core.tools import tool

# ------------------------
# 5. Wrap RAG as a TOOL
# ------------------------
@tool
def rag_tool(question: str) -> str:
    """Use the RAG pipeline (PDF + Web vector store) to answer a question."""
    response = rag_chain.invoke(
        {
            "input": question,
            # you can later pass history if you want:
            "chat_history": []
        }
    )
    return response["answer"]

tools = [rag_tool]
tool_node = ToolNode(tools)

# Bind tools to the LLM so it can decide when to call them
llm_with_tools = llm.bind_tools(tools)

# ------------------------
# 6. Agent node: decides what to do next
# ------------------------
def agent_node(state: MessagesState):
    """
    Take the conversation so far, let the LLM decide:
    - respond directly, or
    - call a tool like rag_tool and keep the answer concise
    """
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    # result can be an AIMessage or a tool call
    return {"messages": [result]}

# ------------------------
# 7. Build LangGraph with Agent + Tools
# ------------------------
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Start → agent
workflow.add_edge(START, "agent")

# Agent can either:
# - go to tools (if it requested a tool call), or
# - end (if it just answered)
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # prebuilt helper from langgraph.prebuilt
)

# After tools run, go back to the agent
workflow.add_edge("tools", "agent")

# Memory for multi-turn chat
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
    if isinstance(event, HumanMessage):
        # Show user messages
        text = (event.content or "").strip()
        if text:
            st.chat_message("user").write(text)

    elif isinstance(event, AIMessage):
        # Skip tool-call messages (no user-facing text)
        if getattr(event, "tool_calls", None):
            continue

        text = (event.content or "").strip()
        if not text:
            # Nothing to speak or show
            continue

        # Show assistant message
        st.chat_message("assistant").write(text)

        # Convert only non-empty text to speech
        tts = gTTS(text, lang="en")
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")






