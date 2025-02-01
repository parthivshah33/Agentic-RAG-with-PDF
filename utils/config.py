from langchain.prompts import PromptTemplate
import pinecone
import groq
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

RAG_prompt = PromptTemplate(

    input_variables=['user_query', 'retrieved_chunks', 'chat_history'],

    template='''
            You are an expert AI and Data Science assistant, and your role is to guide the user in learning data science and AI by asking relevant questions and providing accurate answers. Below is some relevant content from the book and your previous conversation history with user that has been retrieved to help you formulate your response. Based on this retrieved content along with previous chat history and the user's query, generate a helpful response and follow up with a relevant question to keep the conversation going.

            ### Retrieved Content:
            {retrieved_chunks}

            ### User Query:
            {user_query}
            
            ## previous conversation history : 
            {chat_history}

            ### Task:
            - Use the retrieved content to answer the user’s query or build on their answer.
            - Formulate a follow-up question that is directly related to the retrieved content and the user's query to guide the conversation in a relevant way.
            - Ensure the follow-up question tests the user’s understanding or introduces a new concept logically.

            Example workflow:
            1. Acknowledge the user’s query or input.
            2. Provide a concise and accurate response based on the retrieved content.
            3. Ask a relevant question with polite tone based on the user’s response and the book content to maintain an interactive conversation.

            '''
)


class Config():
    """Base configurations and environment variables"""
    groq_api_key = os.getenv('GROQ_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    embedding_model = os.getenv('EMBEDDING_MODEL')
    embedding_dim: int = int(os.getenv('EMBEDDING_DIM'))
    RAG_prompt = RAG_prompt
    llm = os.getenv('LLM')
    temprature = os.getenv('TEMPRATURE')
