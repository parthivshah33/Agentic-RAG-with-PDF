import os, re, random, string
from tqdm import tqdm
from utils.config import Config as config
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec, exceptions


class EmbeddingPipeline:
    def __init__(self):
        """Initialize the EmbeddingPipeline with necessary configurations."""
        self.gemini_api_key = config.gemini_api_key
        self.embedding_model = config.embedding_model
        self.embedding_dim = config.embedding_dim
        self.embedding_client = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model, task_type="retrieval_document", google_api_key=self.gemini_api_key)

    def create_pinecone_index(self, index_name, dimension):
        """Create a Pinecone index if it doesn't already exist."""
        pc = Pinecone(api_key=config.pinecone_api_key)
        try:
            if index_name not in pc.list_indexes():
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Index '{index_name}' created successfully!")
            else:
                print(f"Index '{index_name}' already exists.")
        except exceptions.PineconeException as e:
            print(f"Failed to create index '{index_name}': {e}\n **May be index name has already taken.")
            random_digits = f"{random.randint(0, 9)}{random.randint(0, 9)}"
            new_index_name = f"{index_name}-{random_digits}"
            print(f"Randomely generated index name : ", new_index_name)
            if len(new_index_name) > 45:
                new_index_name = new_index_name[:43] + str(random.randint(10, 99))  #Trims the input string to a maximum length of 43 characters if it exceeds 45 characters, and appends two random digits at the end.
            self.create_pinecone_index(new_index_name, dimension)
            
    def to_kebab_case(self, text):
        """Convert a string to kebab case."""
        text = re.sub(r'[\s_]+', '-', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1-\2', text)
        text = re.sub(r'[^a-zA-Z0-9\-]', '', text)
        return text.lower()

    def load_pdf(self, file_path: str):
        """Load PDF file and return the documents."""
        baseFileName = os.path.basename(file_path)
        print(baseFileName)
        file_name, file_extension = os.path.splitext(baseFileName)
        if file_extension == ".pdf":
            documents = PyPDFLoader(file_path).load()
            return documents
        else:
            raise ValueError("The selected file type is not supported")

    def prepare_and_inject_embeddings(self, documents, index_name):
        """Prepare documents for data injection and generate embeddings."""
        chunk_size = 400
        chunk_overlap = 20
        length_function = len
        is_separator_regex = False

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex
        )

        doc_chunks = text_splitter.split_documents(documents)
        print(f"--- Total : {len(doc_chunks)} Injecting to Vector Database---")

        vectorstore_from_docs = PineconeVectorStore.from_documents(doc_chunks,
                                                                   index_name=index_name,
                                                                   embedding=self.embedding_client)

    def run_pipeline(self, file_path):
        """Main pipeline to load PDF, prepare and inject embeddings to Pinecone server."""
        baseFileName = os.path.basename(file_path)
        index_name, file_extension = os.path.splitext(baseFileName)
        index_name = self.to_kebab_case(index_name)
        print(f"Extracted Index Name : ", index_name)
        self.create_pinecone_index(index_name, self.embedding_dim)
        documents = self.load_pdf(file_path=file_path)
        print(" === documents loaded from pdf === ")

        self.prepare_and_inject_embeddings(
            documents=documents, index_name=index_name)
        print(" _____ embeddings generated and Injected to PINECONE Server ______ ")
        
        return index_name


if __name__ == "__main__":
    file_path = r"D:\freelance\projects\Agentic RAG Chatbot\knowledge documents\machine_learning_tutorial.pdf"
    pipeline = EmbeddingPipeline()
    pipeline.run_pipeline(file_path)