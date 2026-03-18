import streamlit as st
import os
from gtts import gTTS

# Set API keys
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="PDF & Webpage Q&A", layout="centered")
st.title("RAG Application")

# Upload PDFs
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Enter URLs
urls_text = st.text_area("Enter webpage URLs (one per line)")
urls = [url.strip() for url in urls_text.split("\n") if url.strip()]

# -----------------------------
# Load documents
# -----------------------------
all_docs = []

for pdf_file in pdf_files:
    with open(f"temp_{pdf_file.name}", "wb") as f:
        f.write(pdf_file.getvalue())
    loader = PyPDFLoader(f"temp_{pdf_file.name}")
    docs = loader.load()
    all_docs.extend(docs)

for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    all_docs.extend(docs)

st.write(f"Total documents loaded: {len(all_docs)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
all_chunks = []
for doc in all_docs:
    all_chunks.extend(text_splitter.split_text(doc.page_content))

st.write(f"Total chunks: {len(all_chunks)}")

# Embeddings and Pinecone index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
index_name = "rag-app-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=st.secrets["PINECONE_ENVIRONMENT"])
    )

index = pc.Index(index_name)

vectorstore = PineconeVectorStore.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    index_name=index_name
)

st.success("All chunks converted into embeddings and stored in Pinecone")

# -----------------------------
# LLM and RAG setup
# -----------------------------
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])
retriever = vectorstore.as_retriever()

# Prompt to rewrite questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.
Convert user's latest question into a clear, standalone query for document retrieval.
Use chat history to understand context. Do NOT answer here.
Output only the rewritten query.
Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY the provided context.

Guidelines:
1. Use only the information from the context.
2. Do NOT make up or assume any facts.
3. If the answer is not in the context, say:
   "I don’t have enough information in the provided documents to answer that."
4. Keep the answer clear, concise, and well-structured.
5. If helpful, use bullet points or short explanations.
6. Maintain a professional and neutral tone.
7. If multiple sources are present, combine them logically.

Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

# -----------------------------
# Tool wrapper for RAG
# -----------------------------
@tool
def rag_tool(question: str, chat_history_serialized: list) -> str:
    """
    Use the RAG pipeline (PDF + Web vector store) to answer user questions.
    chat_history_serialized: list of dicts with {"role": "user/assistant/system", "content": "..."}
    """
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history_serialized
    })
    return response["answer"]

tools = [rag_tool]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# Agent node
# -----------------------------
def agent_node(state: MessagesState):
    messages = state["messages"]

    # Serialize messages for RAG
    chat_history_serialized = []
    for m in messages:
        if isinstance(m, HumanMessage):
            chat_history_serialized.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            chat_history_serialized.append({"role": "assistant", "content": m.content})
        else:
            chat_history_serialized.append({"role": "system", "content": m.content})

    # LLM with tools
    result = llm_with_tools.invoke(messages)

    # Inject serialized chat history if tool call
    if isinstance(result, AIMessage) and getattr(result, "tool_calls", None):
        for tool_call in result.tool_calls:
            tool_call["args"]["chat_history_serialized"] = chat_history_serialized

    return {"messages": [result]}

# -----------------------------
# Build LangGraph workflow
# -----------------------------
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# Streamlit chat
# -----------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat1"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask a question...")
if user_input:
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)
    st.chat_message("user").write(user_input)

    # Serialize chat history
    chat_history_serialized = []
    for m in st.session_state["messages"]:
        if isinstance(m, HumanMessage):
            chat_history_serialized.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            chat_history_serialized.append({"role": "assistant", "content": m.content})

    events = app.invoke(
        {"messages": st.session_state["messages"]},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
    )

    for event in events["messages"]:
        if isinstance(event, AIMessage):
            st.session_state["messages"].append(event)
            st.chat_message("assistant").write(event.content)

            # TTS
            tts = gTTS(event.content, lang="en")
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
