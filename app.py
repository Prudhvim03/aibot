# Indian Farming AI Assistant - End-to-End Production Code
# Created by Prudhvi

import os
import logging
from typing import Optional, Dict, Any, List
from PIL import Image
import streamlit as st

# --- Import AI Frameworks (Replace with your actual wrappers/APIs) ---
from crewai import Crew, Agent, Task, Process
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import GroqLLM  # Hypothetical, replace with actual Groq LLM wrapper
from tavily import TavilyClient     # Hypothetical, replace with actual Tavily API wrapper
from livekit import LiveKitClient   # Hypothetical, replace with actual LiveKit API wrapper

# --- CONFIGURATION & INITIALIZATION ---

# Environment variables for API keys and endpoints
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Embeddings and Vector DB
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = Qdrant(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embeddings=embeddings,
    collection_name="farming_knowledge"
)

# Initialize LLM, Web Search, and Voice APIs
llm = GroqLLM(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)
livekit = LiveKitClient(api_key=LIVEKIT_API_KEY)

# --- AGENT DEFINITIONS ---

class RouterAgent(Agent):
    """Routes the user query to the appropriate agent based on content."""
    def act(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        if context and context.get("image"):
            return "image_analysis"
        if any(word in query.lower() for word in ["image", "disease", "leaf", "spot", "upload"]):
            return "image_analysis"
        return "text_query"

class RetrieverAgent(Agent):
    """Retrieves relevant documents from vector DB and web."""
    def act(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        docs = vector_db.similarity_search(query, k=3)
        web_results = tavily.search(query)
        return {"docs": docs, "web_results": web_results}

class GraderAgent(Agent):
    """Filters and grades retrieved documents for relevance."""
    def act(self, retrieved: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filtered_docs = [doc for doc in retrieved["docs"] if len(doc.page_content) > 50]
        return {"docs": filtered_docs, "web_results": retrieved["web_results"]}

class HallucinationGrader(Agent):
    """Detects hallucinations in the LLM's answer."""
    def act(self, answer: str, context: Optional[Dict[str, Any]] = None) -> bool:
        # Placeholder: Replace with a robust hallucination detection model
        if "not sure" in answer.lower() or "cannot" in answer.lower():
            return False
        return True

class AnswerGrader(Agent):
    """Ensures the answer is clear, complete, and actionable."""
    def act(self, answer: str, context: Optional[Dict[str, Any]] = None) -> bool:
        return len(answer.split()) > 10

class ImageAnalyzerAgent(Agent):
    """Analyzes uploaded crop images for diseases or issues."""
    def act(self, image: Image.Image, context: Optional[Dict[str, Any]] = None) -> str:
        # Placeholder: Integrate with a real vision model for production
        return "Detected possible leaf blight. Suggest spraying copper-based fungicide."

# --- TASK DEFINITIONS ---

def router_task(query: str, context: Dict[str, Any]) -> str:
    return RouterAgent().act(query, context)

def retriever_task(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return RetrieverAgent().act(query, context)

def grader_task(retrieved: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return GraderAgent().act(retrieved, context)

def hallucination_task(answer: str, context: Dict[str, Any]) -> bool:
    return HallucinationGrader().act(answer, context)

def answer_task(query: str, context: Dict[str, Any]) -> str:
    if context.get("route") == "image_analysis":
        return ImageAnalyzerAgent().act(context.get("image"), context)
    docs = context.get("graded", {}).get("docs", [])
    web = context.get("graded", {}).get("web_results", [])
    context_text = " ".join([doc.page_content for doc in docs])
    web_text = " ".join([res['snippet'] for res in web])
    prompt = (
        f"You are a personal Indian farming support agent. "
        f"User's question: {query}\n\n"
        f"Knowledge base: {context_text}\n\n"
        f"Web search: {web_text}\n\n"
        f"Give a clear, actionable, and friendly answer in simple language."
    )
    return llm.generate(prompt)

# --- CREW DEFINITION ---

def build_crew() -> Crew:
    return Crew(
        agents=[
            RouterAgent(),
            RetrieverAgent(),
            GraderAgent(),
            HallucinationGrader(),
            AnswerGrader(),
            ImageAnalyzerAgent(),
        ],
        tasks=[
            router_task,
            retriever_task,
            grader_task,
            hallucination_task,
            answer_task
        ],
        process=Process.sequential,
        verbose=True,
        cache=True,
        planning=True
    )

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Indian Farming AI Assistant", layout="wide")
st.title("🌾 Indian Farming AI Assistant")
st.caption("Created by Prudhvi")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

crew = build_crew()

def handle_query(query: str, image: Optional[Image.Image] = None) -> str:
    context: Dict[str, Any] = {}
    if image:
        context["image"] = image
    context["route"] = router_task(query, context)
    if context["route"] == "image_analysis":
        answer = answer_task(query, context)
    else:
        retrieved = retriever_task(query, context)
        graded = grader_task(retrieved, context)
        context["graded"] = graded
        answer = answer_task(query, context)
    # Hallucination and grading
    if not hallucination_task(answer, context):
        answer = "Sorry, I am not confident in my answer. Please consult a local expert."
    elif not AnswerGrader().act(answer, context):
        answer = "I need more information to provide a complete answer."
    return answer

# --- MAIN CHAT INTERFACE ---

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your farming question (voice or text):", "")
    image_file = st.file_uploader("Upload crop image (optional)", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Send")

if submit and (user_input or image_file):
    img = None
    if image_file:
        img = Image.open(image_file)
    answer = handle_query(user_input, img)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("agent", answer))

# --- DISPLAY CHAT HISTORY ---

for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**👨‍🌾 You:** {text}")
    else:
        st.markdown(f"**🤖 Agent:** {text}")

# --- LIVEKIT VOICE CHAT (Placeholder) ---

if st.button("🎤 Start Voice Chat"):
    st.info("Voice chat feature coming soon. (Integrate LiveKit API here)")

st.caption("Powered by CrewAI, LangChain, Groq, Tavily, LiveKit. For Indian farmers.")

# --- END OF FILE ---
