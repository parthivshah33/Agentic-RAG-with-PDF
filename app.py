import streamlit as st
import time
from src.agentic_rag import AgenticRag


# Streamed response emulator
def process_chat_request(user_query):
    print(f"User Query: {user_query}")
    agent = AgenticRag()
    chat_response = agent.run(user_query=user_query)
    response = chat_response[-1].content
    # print(response)
    # return response
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    

# result = process_chat_request("What is Machine Learning?")
# print(result)

st.title("AI PDF Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(process_chat_request(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})