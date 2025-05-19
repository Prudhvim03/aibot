import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Indian Farming AI Assistant", layout="wide")
st.title("🌾 Indian Farming AI Assistant")
st.caption("Created by Prudhvi")

@st.cache_resource(show_spinner="Loading vector database...")
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
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2)

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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def analyze_image(image: Image.Image) -> str:
    return "Image analysis is not yet implemented. Please describe your crop issue in text."

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your farming question:", "")
    image_file = st.file_uploader("Or upload a crop image (optional)", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Send")

if submit and (user_input or image_file):
    if image_file:
        img = Image.open(image_file)
        answer = analyze_image(img)
        st.session_state.chat_history.append(("user", "[Image uploaded]"))
        st.session_state.chat_history.append(("agent", answer))
    if user_input:
        result = qa_chain({"query": user_input})
        answer = result["result"]
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("agent", answer))

for role, text in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**👨‍🌾 You:** {text}")
    else:
        st.markdown(f"**🤖 Agent:** {text}")

st.info("This assistant uses a knowledge base via Qdrant and OpenAI for Indian farming support.")
st.caption("For best results, describe your crop or farming issue in detail.")
