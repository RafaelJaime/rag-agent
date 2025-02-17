from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Here is why http://sbert.net/docs/sentence_transformer/pretrained_models.html im using this model
HUG_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
documents_path = "./knowledge_base/"
db_name = "ecuatorian_db"

class DataLoader:

    @staticmethod
    def _load_pdf(path):
        loader = PyPDFLoader(documents_path + path)
        pages = loader.load()
        return pages

    @staticmethod
    def _index_data(pages):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        return chunks

    @staticmethod
    def load_ecuatorian_info():
        """Load ecuatorian tariff from knowledge_base and split it.

        :return: In Memory Vector Store from langchain lib.
        :rtype: InMemoryVectorStore
        """
        pages = DataLoader._load_pdf("tariff/output.pdf")
        chunks = DataLoader._index_data(pages)
        return chunks
    
    @staticmethod
    def load_document(document_path, country_name=None):
        """Load a document from the specified path and split it.
        
        :param document_path: Path to the document file
        :param country_name: Name of the country (optional metadata)
        :return: List of document chunks
        :rtype: list
        """
        if document_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(document_path)
        else:
            loader = TextLoader(document_path, encoding='utf-8')
        
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        if country_name:
            for chunk in chunks:
                chunk.metadata['country'] = country_name
        
        return chunks

class VectorStore:

    @staticmethod
    def get_in_memory_vector_store() -> InMemoryVectorStore:
        """Create a Vector Store using HuggingFace Embeddings and in memory vector store from langchain.

        :return: In Memory Vector Store from langchain lib.
        :rtype: InMemoryVectorStore
        """
        embeddings = HuggingFaceEmbeddings(model_name=HUG_MODEL)
        vectorstore = InMemoryVectorStore(embeddings)
        vectorstore.add_documents(DataLoader.load_ecuatorian_info())
        return vectorstore

    @staticmethod
    def get_chroma_vector_store(reload: bool = False) -> Chroma:
        """
        Create a Vector Store using Chroma from langchain and OpenAIEmbeddings.
        This requires having `OPENAI_API_KEY` in the environment.

        :param reload: Boolean flag to reload (delete and create again) the store (optional)
        :return: Chroma vector store integration from Langchain_chroma library.
        :rtype: Chroma
        """
        embeddings = OpenAIEmbeddings()
        if not os.path.exists(db_name) or reload:
            if reload:
                Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
            return Chroma.from_documents(
                documents=DataLoader.load_ecuatorian_info(),
                embedding=embeddings,
                persist_directory=db_name
            )
        return Chroma(persist_directory=db_name, embedding_function=embeddings)

    @staticmethod
    def create_or_load_qdrant_vector_store(document_paths, country_name, collection_name=None):
        """
        Create or load a Qdrant Vector Store for specific country documents.

        :param document_paths: List of paths to the document files
        :param country_name: Name of the country
        :param collection_name: Name for the Qdrant collection (optional)
        :return: Qdrant vector store instance
        """
        try:
            client = QdrantClient(url="http://localhost:6333")
            collection_name = collection_name if collection_name is not None else f"tariff_{country_name.lower()}"

            collections = client.get_collections()
            collection_exists = any(collection.name == collection_name for collection in collections.collections)

            if collection_exists:
                return QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embedding=OpenAIEmbeddings()
                )

            all_docs = []
            for doc_path in document_paths:
                country_docs = DataLoader.load_document(doc_path, country_name)
                all_docs.extend(country_docs)

            return QdrantVectorStore.from_documents(
                documents=all_docs,
                embedding=OpenAIEmbeddings(),
                collection_name=collection_name,
                url="http://localhost:6333",
            )
        except Exception as e:
            print(f"Error handling vector store for {country_name}: {str(e)}")
            raise
