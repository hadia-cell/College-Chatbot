import os
import re
from pathlib import Path
from typing import TypedDict, Annotated
 
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 
load_dotenv()
BASE_DIR = Path(__file__).parent
 
# ------------------------------------------------------------------ #
#  PAGE CONFIG + CSS
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="IITM Campus Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)
 
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 820px; }
 
/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    border-radius: 22px;
    padding: 28px 30px;
    color: white;
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.30);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.10);
}
.hero h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px; color: white; }
.hero p  { margin: 6px 0 0 0; opacity: .9; font-size: .95rem; }
.badge {
    display: inline-block; margin-top: 14px; padding: 5px 14px;
    background: rgba(255,255,255,0.20); backdrop-filter: blur(6px);
    border-radius: 999px; font-size: .78rem; font-weight: 600;
}
 
/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.06);
    animation: fadeUp .35s ease;
}
@keyframes fadeUp { from {opacity:0; transform: translateY(8px);} to {opacity:1; transform:none;} }
 
/* ---------- Topic tag ---------- */
.tag {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .72rem; font-weight: 600; margin-bottom: 8px;
}
.tag-academic { background: rgba(79,70,229,.15);  color: #6366f1; }
.tag-fee      { background: rgba(16,185,129,.15); color: #10b981; }
.tag-general  { background: rgba(245,158,11,.15); color: #f59e0b; }
 
/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.18); }
.side-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
.side-sub   { opacity: .65; font-size: .8rem; margin-bottom: 14px; }
.stButton > button {
    width: 100%; border-radius: 12px; text-align: left;
    border: 1px solid rgba(128,128,128,0.25);
    transition: all .2s ease; font-size: .85rem;
}
.stButton > button:hover {
    border-color: #7c3aed; color: #7c3aed;
    transform: translateX(3px); box-shadow: 0 4px 14px rgba(124,58,237,.15);
}
 
/* ---------- Empty state ---------- */
.empty {
    text-align: center; padding: 38px 10px; opacity: .85;
}
.empty .emoji { font-size: 3rem; }
.empty h3 { margin: 8px 0 4px 0; }
.empty p  { opacity: .65; font-size: .92rem; }
</style>
""",
    unsafe_allow_html=True,
)
 
# ------------------------------------------------------------------ #
#  BACKEND (cached so PDFs are indexed only once)
# ------------------------------------------------------------------ #
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str
 
 
def clean(text: str) -> str:
    """Remove <think> blocks that reasoning models (e.g. Qwen) may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
 
 
@st.cache_resource(show_spinner="📚 Reading college documents...")
def load_app():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
    def build_retriever(pdf_name: str):
        docs = PyPDFLoader(str(BASE_DIR / pdf_name)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        return FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 4})
 
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
 
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"), temperature=0.4)
 
    def classifier_node(state: State) -> dict:
        q = state["messages"][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {q}\n\n"
            "Return only one word: academic, fee, or general."
        )
        category = clean(llm.invoke(prompt).content).lower()
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"query_type": category}
 
    def academic_rag_node(state: State) -> dict:
        docs = academic_retriever.invoke(state["messages"][-1].content)
        return {"retrieved_context": "\n\n".join(d.page_content for d in docs)}
 
    def fee_rag_node(state: State) -> dict:
        docs = fee_retriever.invoke(state["messages"][-1].content)
        return {"retrieved_context": "\n\n".join(d.page_content for d in docs)}
 
    def general_node(state: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}
 
    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]
 
        # last few turns for conversational memory
        history = ""
        for m in state["messages"][:-1][-6:]:
            role = "Student" if m.type == "human" else "Assistant"
            history += f"{role}: {m.content}\n"
 
        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly and helpful college assistant talking to a {programme} student. "
                f"Answer using your general knowledge. Keep it short and warm.\n\n"
                f"Conversation so far:\n{history}\n"
                f"Question: {query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible. "
                f"If the answer is not in the context, say so honestly.\n\n"
                f"Context:\n{context}\n\n"
                f"Conversation so far:\n{history}\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer. Use markdown (bullets, bold) when helpful."
            )
        return {"messages": [("ai", clean(llm.invoke(prompt).content))]}
 
    def route_query(state: State):
        return {"academic": "academic_rag", "fee": "fee_rag"}.get(state["query_type"], "general")
 
    g = StateGraph(State)
    g.add_node("classifier", classifier_node)
    g.add_node("academic_rag", academic_rag_node)
    g.add_node("fee_rag", fee_rag_node)
    g.add_node("general", general_node)
    g.add_node("response", response_node)
    g.add_edge(START, "classifier")
    g.add_conditional_edges("classifier", route_query)
    g.add_edge("academic_rag", "response")
    g.add_edge("fee_rag", "response")
    g.add_edge("general", "response")
    g.add_edge("response", END)
    return g.compile()
 
 
# ------------------------------------------------------------------ #
#  SESSION STATE
# ------------------------------------------------------------------ #
if "history" not in st.session_state:
    st.session_state.history = []   # {"role", "content", "tag"}
if "pending" not in st.session_state:
    st.session_state.pending = None
 
TAGS = {
    "academic": ("📘 Academics", "tag-academic"),
    "fee": ("💰 Fees", "tag-fee"),
    "general": ("💬 General", "tag-general"),
}
 
# ------------------------------------------------------------------ #
#  SIDEBAR
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown('<div class="side-title">🎓 IITM Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Your 24×7 campus guide</div>', unsafe_allow_html=True)
 
    programme = st.selectbox("Your programme", ["BCA", "BBA", "B.Com (H)"])
 
    st.markdown("---")
    st.markdown("**✨ Try asking**")
    suggestions = {
        "📅 What is the attendance policy?": "What is the minimum attendance required?",
        "💸 What's my annual fee?": "What is my total annual fee for each year?",
        "⏰ Late fee penalty?": "What are the late payment charges?",
        "🏆 Scholarships available?": "Which scholarships can I apply for?",
        "📝 Marking scheme": "What is the marking scheme for theory and practical papers?",
        "🚀 Promotion rules": "What are the rules for promotion to the next year?",
    }
    for label, q in suggestions.items():
        if st.button(label, key=label):
            st.session_state.pending = q
 
    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()
 
    st.caption("Answers come from the official 2024-25 Handbook & Fee Structure.")
 
# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
st.markdown(
    f"""
<div class="hero">
  <h1>🎓 IITM Campus Assistant</h1>
  <p>Ask anything about academics, exams, fees and scholarships.</p>
  <span class="badge">👤 {programme} Student</span>
</div>
""",
    unsafe_allow_html=True,
)
 
try:
    app = load_app()
except Exception as e:
    st.error(f"Could not start the assistant: {e}")
    st.info("Check that `academics_handbook.pdf`, `fee_structure.pdf` and your `GROQ_API_KEY` in `.env` are in place.")
    st.stop()
 
# empty state
if not st.session_state.history:
    st.markdown(
        """
<div class="empty">
  <div class="emoji">👋</div>
  <h3>Hi! How can I help you today?</h3>
  <p>Pick a suggestion from the sidebar or type your question below.</p>
</div>
""",
        unsafe_allow_html=True,
    )
 
# render history
for msg in st.session_state.history:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("tag"):
            label, css = TAGS[msg["tag"]]
            st.markdown(f'<span class="tag {css}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
 
# input
user_query = st.chat_input("Ask about attendance, fees, exams, scholarships...")
if st.session_state.pending:
    user_query = st.session_state.pending
    st.session_state.pending = None
 
if user_query:
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)
 
    # build message list with previous turns for memory
    msgs = [(m["role"] if m["role"] == "user" else "ai", m["content"]) for m in st.session_state.history]
    msgs = [("human" if r == "user" else r, c) for r, c in msgs] + [("human", user_query)]
 
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({"programme": programme, "messages": msgs})
                answer = result["messages"][-1].content
                tag = result.get("query_type", "general")
            except Exception as e:
                answer, tag = f"⚠️ Something went wrong: `{e}`", "general"
        label, css = TAGS[tag]
        st.markdown(f'<span class="tag {css}">{label}</span>', unsafe_allow_html=True)
        st.markdown(answer)
 
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": answer, "tag": tag})