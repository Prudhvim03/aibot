# app.py
# Indian Farming AI Assistant
# Created by Prudhvi

import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env
load_dotenv()

# Import AI frameworks
from crewai import Crew, Agent, Task, Process
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI  # Use OpenAI as placeholder for Groq
import requests

# --- CONFIGURATION ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- EMBEDDINGS & VECTOR DB SETUP ---

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = Qdrant(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embeddings=embeddings,
    collection_name="farming_knowledge"
)

# --- LLM SETUP (replace with Groq/OpenAI as needed) ---

llm = OpenAI(api_key=GROQ_API_KEY)  # Replace with GroqLLM if available

# --- AGENT DEFINITIONS ---

class RouterAgent(Agent):
    def act(self, query, context=None):
        if context and context.get("image"):
            return "image_analysis"
        if any(word in query.lower() for word in ["image", "disease", "leaf", "spot", "upload"]):
            return "image_analysis"
        return "text_query"

class RetrieverAgent(Agent):
    def act(self, query, context=None):
        docs = vector_db.similarity_search(query, k=3)
        # Placeholder for Tavily web search
        web_results = []
        return {"docs": docs, "web_results": web_results}

class GraderAgent(Agent):
    def act(self, retrieved, context=None):
        filtered_docs = [doc for doc in retrieved["docs"] if len(doc.page_content) > 50]
        return {"docs": filtered_docs, "web_results": retrieved["web_results"]}

class HallucinationGrader(Agent):
    def act(self, answer, context=None):
        if "not sure" in answer.lower() or "cannot" in answer.lower():
            return False
        return True

class AnswerGrader(Agent):
    def act(self, answer, context=None):
        return len(answer.split()) > 10

class ImageAnalyzerAgent(Agent):
    def act(self, image, context=None):
        # Placeholder: Integrate with real vision model for production
        return "Detected possible leaf blight. Suggest spraying copper-based fungicide."

# --- TASK DEFINITIONS ---

def router_task(query, context):
    return RouterAgent().act(query, context)

def retriever_task(query, context):
    return RetrieverAgent().act(query, context)

def grader_task(retrieved, context):
    return GraderAgent().act(retrieved, context)

def hallucination_task(answer, context):
    return HallucinationGrader().act(answer, context)

def answer_task(query, context):
    if context.get("route") == "image_analysis":
        return ImageAnalyzerAgent().act(context.get("image"), context)
    docs = context.get("graded", {}).get("docs", [])
    context_text = " ".join([doc.page_content for doc in docs])
    prompt = (
        f"You are a personal Indian farming support agent. "
        f"User's question: {query}\n\n"
        f"Knowledge base: {context_text}\n\n"
        f"Give a clear, actionable, and friendly answer in simple language."
    )
    return llm(prompt)

# --- CREW DEFINITION ---

def build_crew():
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
        verbose=False,
        cache=True,
        planning=True
    )

# --- STREAMLIT UI ---

st.set_page_config(page_title="Indian Farming AI Assistant", layout="wide")
st.title("🌾 Indian Farming AI Assistant")
st.caption("Created by Prudhvi")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

crew = build_crew()

def handle_query(query, image=None):
    context = {}
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

for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**👨‍🌾 You:** {text}")
    else:
        st.markdown(f"**🤖 Agent:** {text}")

st.caption("Powered by CrewAI, LangChain, Qdrant, and HuggingFace. For Indian farmers.")

# --- END OF FILE ---
