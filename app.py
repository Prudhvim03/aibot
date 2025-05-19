# Deep Indian Farming AI Agent (LangChain RAG + Streamlit)
# Created by Prudhvi

import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI  # Replace with GroqLLM when available

# ====== Load Environment Variables ======
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# ====== Streamlit UI Setup ======
st.set_page_config(page_title="🌾 Indian Farming AI Agent", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #f6fff7; }
    .stTextInput>div>div>input { background-color: #e6f2e6; }
    .stButton>button { background-color: #4caf50; color: white; font-weight: bold; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True
)
st.title("🌾 Deep Indian Farming AI Agent")
st.caption("Your personal, multimodal farming support assistant. Created by Prudhvi.")

# ====== Helper: Get Vector DB ======
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Qdrant(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embeddings=embeddings,
        collection_name="farming_knowledge"
    )
    return db

vector_db = get_vector_db()

# ====== LLM Setup (Groq API Placeholder) ======
llm = OpenAI(openai_api_key=GROQ_API_KEY, temperature=0.2)

# ====== Prompt Template ======
prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are a friendly, expert Indian farming assistant. "
        "Use the context and chat history below to answer the user's question in simple, actionable language.\n"
        "Chat History: {chat_history}\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# ====== Conversational Memory ======
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ====== Tavily Web Search (Placeholder) ======
def tavily_web_search(query: str) -> str:
    # Replace with actual Tavily API call
    return f"[Tavily search results for: {query}]"

# ====== Image Analyzer (Placeholder) ======
def analyze_image(image: Image.Image) -> str:
    # Replace with your vision model or API
    return "Image analysis is coming soon. Please describe your crop issue in text for now."

# ====== Voice Assistant (LiveKit API Placeholders) ======
def handle_voice_input():
    # Integrate LiveKit API for speech-to-text
    return None

def handle_voice_output(text):
    # Integrate LiveKit API for text-to-speech
    pass

# ====== Conversational RAG Chain ======
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# ====== Session State for Chat ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Main Chatbot Interface ======
st.markdown("#### 👩‍🌾 Chat, Voice, and Image Support")
col1, col2 = st.columns([3, 1])

with col1:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question (or use voice):", "")
        image_file = st.file_uploader("Upload a crop image (optional)", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Send")

    # Voice input button (placeholder)
    if st.button("🎤 Speak (Voice Input)", key="voice_in"):
        voice_text = handle_voice_input()
        if voice_text:
            user_input = voice_text
            st.success(f"Voice input recognized: {voice_text}")

    if submit and (user_input or image_file):
        if image_file:
            img = Image.open(image_file)
            diagnosis = analyze_image(img)
            st.session_state.chat_history.append(("user", "[Image uploaded]"))
            st.session_state.chat_history.append(("agent", diagnosis))
        if user_input:
            # Web search trigger (example: keywords)
            web_info = ""
            if any(word in user_input.lower() for word in ["weather", "market", "news", "price"]):
                web_info = tavily_web_search(user_input)
            # RAG-based answer
            result = qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            if web_info:
                answer += f"\n\n🌐 Web info: {web_info}"
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("agent", answer))
            # Voice output (placeholder)
            handle_voice_output(answer)

with col2:
    st.markdown("##### 🤖 Agent Info")
    st.write("""
    - **Conversational memory**
    - **Image and voice ready**
    - **Web search (Tavily)**
    - **Powered by Groq (LLM), Qdrant (RAG)**
    """)

# ====== Display Chat History ======
st.markdown("---")
st.markdown("### Conversation")
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div style='background:#e6f2e6;padding:10px;border-radius:10px;'><b>👨‍🌾 You:</b> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#f0f0f0;padding:10px;border-radius:10px;'><b>🤖 Agent:</b> {text}</div>", unsafe_allow_html=True)

st.info("This assistant uses a knowledge base via Qdrant and Groq/OpenAI for Indian farming support. Voice and image features are modular and ready for integration.")
st.caption("For best results, describe your crop or farming issue in detail. Voice and image analysis coming soon!")

# ====== END OF FILE ======
