from utils.config import Config as config
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


class AgenticRag:
    def __init__(self, pinecone_index_name: str, thread_id):

        self.llm = ChatGroq(temperature=config.temprature,
                            model_name=config.llm,
                            api_key=config.groq_api_key
                            )

        self.embedding_client = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            task_type="retrieval_document",
            google_api_key=config.gemini_api_key)

        self.pinecone_index = Pinecone(
            api_key=config.pinecone_api_key
        ).Index(pinecone_index_name)

        self.vector_store = PineconeVectorStore(
            index=self.pinecone_index, embedding=self.embedding_client)

        self.retriever = self.vector_store.as_retriever()

        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="Document Retriever Tool",
            description="Tool to fetch documents from a database whenever a query is given."
        )

        self.tools = [self.retriever_tool]

        self.workflow = self.create_workflow()

        self.thread_id = thread_id

        self.rag_prompt = config.RAG_prompt
        self.main_agent_prompt = config.main_agent_prompt

    def create_workflow(self):
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        def agent(state):
            print("---CALLING MAIN AGENT---")
            messages = state["messages"]
            prompt = self.main_agent_prompt
            model = self.llm
            model = model.bind_tools(self.tools)
            response = model.invoke(messages)
            return {"messages": [response]}

        def generate(state):
            print("---GENERATE---")
            messages = state["messages"]
            question = messages[-2].content
            docs = messages[-1].content
            prompt = self.rag_prompt
            llm = self.llm
            rag_chain = prompt | llm | StrOutputParser()
            response = rag_chain.invoke(
                {"retrieved_chunks": docs, "user_query": question, "chat_history": messages})
            return {"messages": ("assistant", response)}

        def grade_documents(state) -> Literal["generate"]:
            return 'generate'

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent)
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)
        # workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {
                                       "tools": "retrieve", END: END})
        workflow.add_conditional_edges("retrieve", grade_documents)
        workflow.add_edge("generate", END)
        # workflow.add_edge("rewrite", "agent")
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def run(self, user_query):
        inputs = {"messages": [("user", user_query)]}
        graph_result = self.workflow.invoke(
            inputs, config={"configurable": {"thread_id": self.thread_id}})
        return graph_result["messages"]


if __name__ == "__main__":
    thread_id = "test1"
    agentic_rag = AgenticRag(
        pinecone_index_name="budget-speech-2025", thread_id=thread_id)
    flag = True
    while flag:
        user_query = input("Enter your query: ")
        if user_query == "q":
            flag = False
        else:
            result = agentic_rag.run(user_query)
            print("result : ", result[-1].content + f"{'*'*50}\n\n")

            print(f"Current Graph State : ", str(result) + f"{'*'*50}\n\n")