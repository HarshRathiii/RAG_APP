import streamlit as st
import os
from gtts import gTTS

# ------------------------
# 0. Set API keys
# ------------------------
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
os.environ['LANGSMITH_API_KEY'] = st.secrets['LANGSMITH_API_KEY']
os.environ['LANGSMITH_PROJECT'] = st.secrets['LANGSMITH_PROJECT']
os.environ['LANGSMITH_TRACING'] = "true"
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# ------------------------
# 1. Streamlit Page Setup
# ------------------------
st.set_page_config(page_title="PDF & Webpage Q&A", layout="centered")
st.title("RAG Application")

# ------------------------
# 2. Upload PDFs & URLs
# ------------------------
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
urls_text = st.text_area("Enter webpage URLs (one per line)")
urls = [url.strip() for url in urls_text.split("\n") if url.strip()]

all_docs = []

# Load PDFs
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
for pdf_file in pdf_files:
    with open(f"temp_{pdf_file.name}", "wb") as f:
        f.write(pdf_file.getvalue())
    loader = PyPDFLoader(f"temp_{pdf_file.name}")
    all_docs.extend(loader.load())

# Load URLs
for url in urls:
    loader = WebBaseLoader(url)
    all_docs.extend(loader.load())

st.write(f"Total documents loaded: {len(all_docs)}")

# ------------------------
# 3. Split into chunks
# ------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
for doc in all_docs:
    all_chunks.extend(text_splitter.split_text(doc.page_content))
st.write(f"Total chunks: {len(all_chunks)}")

# ------------------------
# 4. Create embeddings + Pinecone index
# ------------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

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
vectorstore = PineconeVectorStore.from_texts(texts=all_chunks, embedding=embeddings, index_name=index_name)
st.success("All chunks converted into embeddings and stored in Pinecone")

# ------------------------
# 5. Setup LLM + RAG
# ------------------------
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])
retriever = vectorstore.as_retriever()

# Contextualize query prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.
Convert the user's latest question into a clear, standalone query for document retrieval.
Use chat history for context. Do NOT answer yet.""" ),
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
    ("system", """You are an AI assistant for a RAG system.
Answer using ONLY the provided context.
If answer missing, say: "I don’t have enough information in the provided documents to answer that."
Keep answers clear, concise, bullet points allowed.
Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

# ------------------------
# 6. Wrap RAG as tool
# ------------------------
from langchain_core.tools import tool

@tool
def rag_tool(question: str, chat_history: list) -> str:
    """Use the RAG pipeline to answer user questions with context awareness."""

    # Serialize messages ONLY for rag_chain
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
# 7. Agent Node
# ------------------------
def agent_node(state: MessagesState):
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    
    # Inject history into tool calls
    if hasattr(result, "tool_calls") and result.tool_calls:
        for call in result.tool_calls:
            call["args"]["chat_history"] = messages

    return {"messages": [result]}

# ------------------------
# 8. Build LangGraph workflow
# ------------------------
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Memory for multi-turn chat
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def serialize_messages(messages):
    return [
        {"role": "user", "content": m.content} if isinstance(m, HumanMessage)
        else {"role": "assistant", "content": m.content} if isinstance(m, AIMessage)
        else {"role": "system", "content": m.content}
        for m in messages

# ------------------------
# 9. Session state setup
# ------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat1"
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ------------------------
# 10. Streamlit chat loop
# ------------------------
user_input = st.chat_input("Ask a question...")
if user_input:
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)

    # Pass messages as BaseMessage objects (do NOT serialize here)
    serialized_msgs = serialize_messages(st.session_state["messages"])
    
    events = app.invoke(
        {"messages": serialized_msgs},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}}
    )

    for event in events["messages"]:
        if event["role"] == "assistant":
            ai_msg = AIMessage(content=event["content"])
            st.session_state["messages"].append(ai_msg)
            st.chat_message("assistant").write(ai_msg.content)
            # TTS
            tts = gTTS(ai_msg.content, lang="en")
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
