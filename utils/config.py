from langchain.prompts import PromptTemplate
import pinecone
import groq
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

main_agent_prompt = PromptTemplate.from_template("""
Name : Alex
You are an expert AI assistant, your role is to chat with user in human like manner and help user find infomation related to any topic.
### Task:
    - You can retrieve document in case you need extra knowledge
    - Behave in friedly but professional way, keep empathy and politeness in your responses.
""")

RAG_prompt = PromptTemplate(

    input_variables=['user_query', 'retrieved_chunks', 'chat_history'],

    template="""
            You are an expert AI assistant, your role is to guide the user by asking relevant questions and providing accurate answers. Below is some relevant content and your previous conversation history with user to help you formulate your response. Based on this retrieved content along with previous chat history and the user's query, generate a helpful response.
            
            ### Retrieved Content:
            {retrieved_chunks}

            ### User Query:
            {user_query}
            
            ## previous conversation history : 
            {chat_history}
            
            ### Task:
            - Use the retrieved content to answer the user’s query.
            - Provide a concise and accurate response based on the retrieved content.
            
            """
)


class Config():
    """Base configurations and environment variables"""
    groq_api_key = os.getenv('GROQ_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    embedding_model = os.getenv('EMBEDDING_MODEL')
    embedding_dim: int = int(os.getenv('EMBEDDING_DIM'))
    RAG_prompt = RAG_prompt
    main_agent_prompt = main_agent_prompt
    llm = os.getenv('LLM')
    temprature = os.getenv('TEMPRATURE')
