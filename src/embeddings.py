import os
from tqdm import tqdm
from utils.config import Config as config
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""load environment variables"""
gemini_api_key = config.gemini_api_key
llama_api_key = config.llamaCloud_api_key
pinecone_index_name = config.pinecone_index_name
'''Initialiazing Embeddings and Chroma Client'''
embedding_client = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", task_type="retrieval_document", google_api_key=gemini_api_key)


def main_pipeline(file_path):
    documents = load_pdf(file_path=file_path)
    print(" === documents loaded from pdf === ")

    prepare_and_inject_embeddings(
        documents=documents, embedding_client=embedding_client)
    print(" _____ embeddings generated and Injected to PINECONE Server ______ ")


def load_pdf(file_path: str):
    """
        Load pdf file and return the documents.

        Args:
            file_directory (str): The directory path of the file to be loaded.

        Returns:
            list of parsed documents.

        Raises:
            ValueError: If the file extension is neither CSV nor Excel.
        """
    baseFileName = os.path.basename(file_path)
    print(baseFileName)
    file_name, file_extension = os.path.splitext(baseFileName)
    if file_extension == ".pdf":
        documents = PyPDFLoader(file_path).load()
        return documents
    else:
        raise ValueError("The selected file type is not supported")


def prepare_and_inject_embeddings(documents, embedding_client):
    """
    -prepare documents for data injection,
    -Generate Embeddigngs along with Injetion to Server.

    Args:
        documents (): Object of Loaded PDF Documents from Docuement Loader Class.
        file_name (str): The base name of the file for use in metadata.

    Returns: None, prints success message after injetion
    """

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
    print(f"--- Total Documents : {len(doc_chunks)} ---")

    vectorstore_from_docs = PineconeVectorStore.from_documents(doc_chunks,
                                                               index_name=pinecone_index_name,
                                                               embedding=embedding_client)

    print("==============================")
    print(" -- Data has been stored in PINECONE Index -- ")


if __name__ == "__main__":
    file_path = r"D:\freelance\projects\Voicebot\knowledge documents\machine_learning_tutorial.pdf"
    main_pipeline(file_path)
