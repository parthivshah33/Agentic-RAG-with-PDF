import os
import time
import uuid
import streamlit as st
from src.agentic_rag import AgenticRag
from src.embeddings import EmbeddingPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="AI PDF Agent",
    page_icon="🔖",
    layout="wide"
)

# Generate a default random UUID (4 digits)
random_session_id = uuid.uuid4().int & ((1 << 16) - 1)  # range: 0 to 65535


# Directory to save uploaded PDFs
SAVE_DIR = "uploaded_pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)


def process_chat_request(user_query, index_name):
    """Handles the AI chat response generation and streaming output."""
    agent = AgenticRag(pinecone_index_name=index_name,
                       thread_id=random_session_id)
    chat_response = agent.run(user_query=user_query)
    response = chat_response[-1].content

    full_response = ""
    for word in response.split():
        full_response += word + " "
        yield word + " "
        time.sleep(0.05)

    st.session_state["last_response"] = full_response


def save_uploaded_pdf(uploaded_file):
    """Saves the uploaded PDF to a local directory."""
    save_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.sidebar.write(f"File saved at: `{save_path}`")
    return save_path


st.title("📄 AI PDF Agent")
st.divider()

# Sidebar for PDF upload
st.sidebar.subheader("Upload Your PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Process PDF only once and store in session state
if uploaded_file:
    # New file uploaded - process and store in session state
    if ('processed_file' not in st.session_state or
            st.session_state.processed_file != uploaded_file.name):

        local_path = save_uploaded_pdf(uploaded_file)
        st.session_state.embedding_index = EmbeddingPipeline().run_pipeline(local_path)
        st.session_state.processed_file = uploaded_file.name
    embedding_index = st.session_state.embedding_index
else:
    # No new file - check existing session state
    if 'processed_file' in st.session_state:
        embedding_index = st.session_state.embedding_index
    else:
        embedding_index = None
        st.sidebar.info("Please upload a PDF file to start.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI on the right side
# chat_container = st.container()

# with chat_container:
st.subheader("💬 AI Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if (prompt := st.chat_input("Ask something about the uploaded PDF or general questions!")) and embedding_index:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = process_chat_request(
            user_query=prompt, index_name=embedding_index)
        st.write_stream(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state["last_response"]})
