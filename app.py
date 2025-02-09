import os
import time
import uuid
import streamlit as st
from src.agentic_rag import AgenticRag
from src.embeddings import EmbeddingPipeline
from streamlit_pdf_viewer import pdf_viewer

# Constants
SAVE_DIR = "uploaded_pdfs"
DEFAULT_INDEX_NAME = "budget-speech"
os.makedirs(SAVE_DIR, exist_ok=True)

def reset_session(new_index, uploaded_file):
    """
    Resets the session state with a new random session ID, new agent (with new index),
    and clears any previous chat messages.
    """
    st.session_state.random_session_id = uuid.uuid4().int & ((1 << 16) - 1)
    st.session_state.agent = AgenticRag(
        pinecone_index_name=new_index,
        thread_id=st.session_state.random_session_id
    )
    st.session_state.embedding_index = new_index
    st.session_state.processed_file = uploaded_file.name
    st.session_state.processed_file_path = os.path.join(SAVE_DIR, uploaded_file.name)  # Store file path
    st.session_state.messages = []  # Clear chat history
    st.sidebar.success("Session re-initialized with the new PDF!")

def save_uploaded_pdf(uploaded_file):
    """Saves the uploaded PDF to a local directory."""
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return save_path

def handle_pdf_upload(uploaded_file):
    """
    Handles the PDF upload. If the PDF is new (i.e., its filename is different
    from the one processed previously), run the embedding pipeline and reset the session.
    """
    if st.session_state.get("processed_file") != uploaded_file.name:
        local_path = save_uploaded_pdf(uploaded_file)
        new_index = EmbeddingPipeline().run_pipeline(local_path)
        reset_session(new_index, uploaded_file)
    return st.session_state.embedding_index

def process_chat_request(user_query):
    """Handles the AI chat response generation and streaming output."""
    agent = st.session_state.agent
    chat_response = agent.run(user_query=user_query)
    response = chat_response[-1].content

    full_response = ""
    for word in response.split():
        full_response += word + " "
        yield word + " "
        time.sleep(0.05)

    st.session_state["last_response"] = full_response

def display_chat_messages():
    """Displays the chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt):
    """Handles the user input and generates a response."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_gen = process_chat_request(user_query=prompt)
        st.write_stream(response_gen)

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state["last_response"]}
    )

# ----------------------------
# INITIAL SESSION INITIALIZATION
# ----------------------------
if "random_session_id" not in st.session_state:
    st.session_state.random_session_id = uuid.uuid4().int & ((1 << 16) - 1)

if "agent" not in st.session_state:
    st.session_state.agent = AgenticRag(
        pinecone_index_name=DEFAULT_INDEX_NAME,
        thread_id=st.session_state.random_session_id
    )

if "embedding_index" not in st.session_state:
    st.session_state.embedding_index = DEFAULT_INDEX_NAME

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "processed_file_path" not in st.session_state:
    st.session_state.processed_file_path = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🤖 Chat with AI Agent")
st.divider()

# Sidebar: PDF Upload
st.sidebar.subheader("Upload Your PDF")
uploaded_file = st.sidebar.file_uploader("Upload PDF here", type=["pdf"])

# Process PDF upload: either use the new PDF or fallback to the default
if uploaded_file:
    embedding_index = handle_pdf_upload(uploaded_file)
else:
    embedding_index = st.session_state.get("embedding_index", DEFAULT_INDEX_NAME)
    if embedding_index == DEFAULT_INDEX_NAME:
        st.sidebar.warning("Upload your own PDF or use the default PDF (Union Budget Speech 2025) for chat.")

# Display uploaded PDF in the sidebar if it exists
if st.session_state.processed_file_path:
    with st.sidebar:
        pdf_viewer(st.session_state.processed_file_path, width=300, height=400)

# Main Chat UI: Display any previous messages
st.subheader("🔍 Ask anything about your PDF! 📚")
display_chat_messages()

if (prompt := st.chat_input("Ask something about the uploaded PDF or general questions!")):
    handle_user_input(prompt)
