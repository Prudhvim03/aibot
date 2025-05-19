# app.py
# Deep Indian Farming AI Agent (CrewAI + LangChain + Streamlit)
# Created by Prudhvi

import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# --- LangChain Components ---
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI  # Replace with GroqLLM when available
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

# --- CrewAI Components ---
from crewai import Agent, Task, Crew, Process

# --- Embeddings and Vector DB ---
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

# --- LLM Setup (Replace with GroqLLM when available) ---
llm = OpenAI(openai_api_key=GROQ_API_KEY, temperature=0.2)

# --- Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful Indian farming assistant. "
        "Use the context below to answer the user's question in simple, actionable language.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

# --- LangChain QA Chain ---
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# --- Tavily Web Search Tool (Placeholder) ---
class TavilySearchTool(BaseTool):
    name = "TavilyWebSearch"
    description = "Searches the web for up-to-date farming information using Tavily API."

    def _run(self, query: str):
        # Replace with actual Tavily API call
        return f"[Tavily search results for: {query}]"

    async def _arun(self, query: str):
        return self._run(query)

# --- Image Analysis Agent (Placeholder) ---
def analyze_image(image: Image.Image) -> str:
    # Replace with actual vision model
    return "Image analysis is not yet implemented. Please describe your crop issue in text."

# --- Voice Assistant Agent (Placeholder) ---
def handle_voice_input():
    # Integrate LiveKit API for speech-to-text
    return None

def handle_voice_output(text):
    # Integrate LiveKit API for text-to-speech
    pass

# --- CrewAI Agents ---
memory = ConversationBufferMemory()

chat_agent = Agent(
    role="Farming Chat Assistant",
    goal="Provide deep, personalized farming support using RAG, web search, and context.",
    backstory="Expert in Indian agriculture, always friendly and clear.",
    tools=[TavilySearchTool()],
    memory=memory,
    llm=llm
)

image_agent = Agent(
    role="Image Analyzer",
    goal="Diagnose crop diseases or issues from images.",
    backstory="Expert in plant pathology.",
    tools=[],
    memory=None,
    llm=llm
)

voice_agent = Agent(
    role="Voice Assistant",
    goal="Enable voice input and output for accessibility.",
    backstory="Helps farmers interact hands-free.",
    tools=[],
    memory=None,
    llm=llm
)

# --- CrewAI Tasks ---
chat_task = Task(
    agent=chat_agent,
    description="Answer farming questions using knowledge base, web search, and context.",
    expected_output="Clear, actionable farming advice."
)

image_task = Task(
    agent=image_agent,
    description="Analyze uploaded crop images for disease or issue detection.",
    expected_output="Diagnosis and suggested action."
)

voice_task = Task(
    agent=voice_agent,
    description="Handle voice input and output for the assistant.",
    expected_output="Transcribed input and spoken output."
)

# --- CrewAI Orchestration ---
crew = Crew(
    agents=[chat_agent, image_agent, voice_agent],
    tasks=[chat_task, image_task, voice_task],
    process=Process.sequential,
    verbose=False
)

# --- Streamlit UI ---
st.set_page_config(page_title="Indian Farming AI Assistant", layout="wide")
st.title("🌾 Indian Farming AI Assistant")
st.caption("Created by Prudhvi")

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
        # Use CrewAI to orchestrate agents
        # For demo, use LangChain QA and Tavily tool directly
        result = qa_chain({"query": user_input})
        answer = result["result"]
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("agent", answer))

# --- Display Chat History ---
for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**👨‍🌾 You:** {text}")
    else:
        st.markdown(f"**🤖 Agent:** {text}")

st.info("This assistant uses CrewAI, LangChain, Qdrant, and (optionally) Groq, Tavily, and LiveKit for Indian farming support.")
st.caption("For best results, describe your crop or farming issue in detail.")

# --- END OF FILE ---
