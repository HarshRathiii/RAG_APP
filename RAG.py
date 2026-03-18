import os
import streamlit as st
from gtts import gTTS

# Set your API keys from Streamlit secrets
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage

# ------------------------
# Streamlit page
# ------------------------
st.set_page_config(page_title="PDF & Web Q&A", layout="centered")
st.title("RAG Application")

# ------------------------
# Step 1: Upload PDFs
# ------------------------
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# ------------------------
# Step 2: Enter URLs
# ------------------------
urls_text = st.text_area("Enter webpage URLs (one per line)")
urls = [u.strip() for u in urls_text.split("\n") if u.strip()]

all_docs = []

# Load PDFs
for pdf_file in pdf_files:
    temp_path = f"temp_{pdf_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())
    loader = PyPDFLoader(temp_path)
    all_docs.extend(loader.load())

# Load webpages
for url in urls:
    loader = WebBaseLoader(url)
    all_docs.extend(loader.load())

st.write(f"Total documents loaded: {len(all_docs)}")

# ------------------------
# Step 3: Split into chunks
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
all_chunks = []
for doc in all_docs:
    all_chunks.extend(text_splitter.split_text(doc.page_content))

st.write(f"Total chunks: {len(all_chunks)}")

# ------------------------
# Step 4: Embeddings & Pinecone
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
index = pc.Index(index_name)

vectorstore = PineconeVectorStore.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    index_name=index_name
)
st.success("All chunks converted into embeddings and stored in Pinecone")

# ------------------------
# Step 5: LLM & RAG chain
# ------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
)

retriever = vectorstore.as_retriever()

# Context-aware query rewriting
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for RAG.
Rewrite the user's latest question into a clear, standalone query for retrieval."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant. Answer the user's question using ONLY the provided context.
If not enough info, respond: "I don’t have enough information in the provided documents to answer that."
Keep it clear and concise."""),
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
# Step 6: Tool wrapper
# ------------------------
from langchain_core.tools import tool

@tool
def rag_tool(question: str, chat_history: list) -> str:
    """Answer question using RAG with context awareness"""
    # Serialize messages
    chat_history_serializable = [
        {"role": "user", "content": m["content"]} if m["role"]=="user" else
        {"role": "assistant", "content": m["content"]} for m in chat_history
    ]
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history_serializable
    })
    return response["answer"]

tools = [rag_tool]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# ------------------------
# Step 7: Agent Node
# ------------------------
def agent_node(state: MessagesState):
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    # Inject history into tool calls
    if isinstance(result, AIMessage) and result.tool_calls:
        for tool_call in result.tool_calls:
            tool_call["args"]["chat_history"] = messages
    return {"messages": [result]}

# ------------------------
# Step 8: LangGraph workflow
# ------------------------
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ------------------------
# Step 9: Multi-turn chat
# ------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat1"
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask a question...")
if user_input:
    # Save user message in JSON-serializable form
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Invoke agent with serialized messages
    events = app.invoke(
        {"messages": st.session_state["messages"]},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
    )

    # Display assistant response and TTS
    for event in events["messages"]:
        content = event.get("content", "")
        role = event.get("role")
        if role == "assistant" and content:
            st.session_state["messages"].append({"role": "assistant", "content": content})
            st.chat_message("assistant").write(content)

            # Only speak if content is non-empty
            if content.strip():
                tts = gTTS(content, lang="en")
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3")
