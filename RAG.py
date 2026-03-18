import os
import streamlit as st
from gtts import gTTS

# ------------------------
# 1. Set environment keys
# ------------------------
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# ------------------------
# 2. Streamlit page setup
# ------------------------
st.set_page_config(page_title="PDF & Webpage Q&A", layout="centered")
st.title("RAG Application")

# ------------------------
# 3. Upload PDFs and URLs
# ------------------------
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# 4. Split text into chunks
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
all_chunks = []
for doc in all_docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)

st.write(f"Total chunks: {len(all_chunks)}")

# ------------------------
# 5. Create embeddings + Pinecone index
# ------------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

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

st.success("All chunks converted into embeddings and stored in Pinecone")

# ------------------------
# 6. LLM + RAG setup
# ------------------------
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
)

retriever = vectorstore.as_retriever()

# ------------------------
# 6a. History-aware retriever for query rewriting
# ------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a RAG system.
Rewrite the user's question into a standalone query using chat history.
Output only the rewritten query."""),
    ("human", "{input}"),
    ("system", "{chat_history}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# ------------------------
# 6b. QA prompt (FIXED: use {context})
# ------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for a RAG system.
Answer the user's question using ONLY the provided context.
If not enough information, respond:
"I don’t have enough information in the provided documents to answer that."

Context:
{context}
"""),
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
# 7. Wrap RAG as a tool
# ------------------------
@tool
def rag_tool(question: str, chat_history: list) -> str:
    """Answer questions using RAG pipeline (PDF + Web)."""
    
    # Serialize messages to dict
    chat_history_serializable = [
        {"role": "user", "content": m.content} if isinstance(m, HumanMessage)
        else {"role": "assistant", "content": m.content} if isinstance(m, AIMessage)
        else {"role": "system", "content": m.content}
        for m in chat_history
    ]
    
    response = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history_serializable
        }
    )
    
    return response["answer"]

tools = [rag_tool]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# ------------------------
# 8. Agent node
# ------------------------
def agent_node(state: MessagesState):
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    
    # Inject history if tool call exists
    if isinstance(result, AIMessage) and getattr(result, "tool_calls", None):
        for tool_call in result.tool_calls:
            tool_call["args"]["chat_history"] = messages
            
    return {"messages": [result]}

# ------------------------
# 9. Build LangGraph
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
# 10. Streamlit chat interface
# ------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat1"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Ask a question...")
if user_input:
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)
    
    # Convert HumanMessage to dict for app.invoke
    messages_serializable = [
        {"role": "user", "content": m.content} if isinstance(m, HumanMessage)
        else {"role": "assistant", "content": m.content} for m in st.session_state["messages"]
    ]
    
    events = app.invoke(
        {"messages": messages_serializable},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
    )
    
    for event in events["messages"]:
        if isinstance(event, dict) and event.get("role") == "assistant":
            ai_msg = AIMessage(content=event.get("content", ""))
            st.session_state["messages"].append(ai_msg)
            
            # Show in Streamlit chat
            st.chat_message("assistant").write(ai_msg.content)
            
            # TTS only if text is non-empty
            if ai_msg.content.strip():
                tts = gTTS(ai_msg.content, lang="en")
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3")
        else:
            # Display user messages
            if isinstance(event, dict) and event.get("role") == "user":
                st.chat_message("user").write(event.get("content", ""))
