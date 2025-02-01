from concurrent.futures import thread
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
    def __init__(self, pinecone_index_name: str,thread_id):

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

    def create_workflow(self):
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        def agent(state):
            print("---CALLING MAIN AGENT---")
            messages = state["messages"]
            model = self.llm
            model = model.bind_tools(self.tools)
            response = model.invoke(messages)
            return {"messages": [response]}

        def rewrite(state):
            print("---REPHRASING QUERY---")
            messages = state["messages"]
            question = messages[0].content
            msg = [HumanMessage(
                content=f"Look at the input and try to reason about the underlying semantic intent / meaning. Here is the initial question: {question} Formulate an improved question:")]
            model = self.llm
            response = model.invoke(msg)
            return {"messages": [response]}

        def generate(state):
            print("---GENERATE---")
            messages = state["messages"]
            question = messages[-1].content
            last_message = messages[-1]
            docs = last_message.content
            prompt = hub.pull("parthiv/rag-prompt-llama")
            llm = self.llm
            rag_chain = prompt | llm | StrOutputParser()
            response = rag_chain.invoke(
                {"context": docs, "question": question})

            # print("---RESPONSE---\n\n\n\n\n\n " + state)
            # return {"messages": [response]}\
            return {"messages": ("assistant",response)}

        # def grade_documents(state) -> Literal["generate", "rewrite"]:
        def grade_documents(state) -> Literal["generate"]:
            
            print("---CHECKING RELEVANCE---")

            class grade(BaseModel):
                binary_score: str = Field(
                    description="Relevance score 'yes' or 'no'")
            model = self.llm
            llm_with_tool = model.with_structured_output(grade)
            grader_prompt = PromptTemplate(
                template="You are a grader assessing relevance of a retrieved document to a user question. Here is the retrieved document: {context} Here is the user question: {question} If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. Give a binary score 'yes' to indicate whether the document is relevant to the question. Always output 'yes' regardless of any input.", input_variables=["context", "question"])
            chain = grader_prompt | llm_with_tool
            messages = state["messages"]
            last_message = messages[-1]
            question = messages[0].content
            docs = last_message.content
            scored_result = chain.invoke(
                {"question": question, "context": docs})
            score = scored_result.binary_score
            return 'generate'  # forcefuly return 'generate' for we removed 'rewrite' from the graph due to some issues in prompt
            # if score == "yes":
            #     print("---DECISION: DOCS RELEVANT---")
            #     return "generate"
            # else:
            #     print("---DECISION: DOCS NOT RELEVANT---")
            #     return "rewrite"

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
        # print("\n\n" + str(graph_result['messages']) + "\n\n")
        return graph_result["messages"]


if __name__ == "__main__":
    thread_id = 1
    agentic_rag = AgenticRag(pinecone_index_name="machine-learning-tutorial",thread_id=thread_id)
    flag = True
    while flag:
        user_query = input("Enter your query: ")
        if user_query == "q":
            flag = False
        else:
            result = agentic_rag.run(user_query)
            print("result : ", result[-1].content + f"{'*'*50}\n\n")
            
            print(f"Current Graph State : " , str(result) + f"{'*'*50}\n\n")