# Indian Farming AI Assistant (LangChain RAG + Streamlit)
# Created by Prudhvi

import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # Replace with GroqLLM wrapper if available
from langchain.chains import ConversationalRetrievalChain

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# --- Streamlit UI setup ---
st.set_page_config(page_title="Indian Farming AI Assistant", layout="wide")
st.title("🌾 Indian Farming AI Assistant")
st.caption("Created by Prudhvi")

# --- Embedding Model and Vector DB ---
@st.cache_resource(show_spinner="Connecting to Knowledge Base...")
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

# --- LLM Setup (replace with GroqLLM when available) ---
llm = OpenAI(openai_api_key=GROQ_API_KEY, temperature=0.2)

# --- Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are a helpful Indian farming assistant. "
        "Use the context and chat history below to answer the user's question in simple, actionable language.\n"
        "Chat History: {chat_history}\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# --- Conversational Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Tavily Web Search Tool (Placeholder) ---
def tavily_web_search(query: str) -> str:
    # Replace with actual Tavily API call
    return f"[Tavily search results for: {query}]"

# --- Image Analysis Agent (Placeholder) ---
def analyze_image(image: Image.Image) -> str:
    # Replace with actual vision model
    return "Image analysis is not yet implemented. Please describe your crop issue in text."

# --- Voice Assistant Placeholders ---
def handle_voice_input():
    # Integrate LiveKit API for speech-to-text
    return None

def handle_voice_output(text):
    # Integrate LiveKit API for text-to-speech
    pass

# --- Conversational RAG Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# --- Session State for Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Main Chat Interface ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your farming question (voice or text):", "")
    image_file = st.file_uploader("Or upload a crop image (optional)", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Send")

if submit and (user_input or image_file):
    if image_file:
        img = Image.open(image_file)
        diagnosis = analyze_image(img)
        st.session_state.chat_history.append(("user", "[Image uploaded]"))
        st.session_state.chat_history.append(("agent", diagnosis))
    if user_input:
        # Use Tavily for web search if needed (example: if "weather" in query)
        web_info = ""
        if any(word in user_input.lower() for word in ["weather", "market", "news"]):
            web_info = tavily_web_search(user_input)
        # RAG-based answer
        result = qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
        answer = result["answer"]
        # Optionally append web info
        if web_info:
            answer += f"\n\n[Web info]: {web_info}"
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("agent", answer))
        # Voice output (placeholder)
        handle_voice_output(answer)

# --- Display Chat History ---
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**👨‍🌾 You:** {text}")
    else:
        st.markdown(f"**🤖 Agent:** {text}")

st.info("This assistant uses a knowledge base via Qdrant and Groq/OpenAI for Indian farming support.")
st.caption("For best results, describe your crop or farming issue in detail. Voice and image features are modular and ready for integration.")

# --- END OF FILE ---
