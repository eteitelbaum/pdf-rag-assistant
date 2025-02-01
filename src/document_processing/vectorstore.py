from typing import List, Optional
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

class VectorStoreManager:
    """Manages vector database operations and embeddings"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./academic_db"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to store the vector database
        """
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model
        )
        self.persist_directory = persist_directory
    
    def initialize_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Initialize or load vector store with documents.
        
        Args:
            documents: List of document chunks to store
            
        Returns:
            Chroma: Initialized vector store
        """
        print("\nGenerating embeddings and storing in database...")
        try:
            # Show progress for embedding generation
            with tqdm(total=len(documents), desc="Generating embeddings", unit="chunk") as pbar:
                def progress_callback(current, total):
                    pbar.update(1)
                
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    progress_callback=progress_callback
                )
            print("\nVector database successfully created and stored!")
            return vectorstore
        except Exception as e:
            print(f"\nError creating vector database: {e}")
            raise
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store if available.
        
        Returns:
            Optional[Chroma]: Loaded vector store or None if not found
        """
        try:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except Exception:
            return None
    
    def get_or_create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Load existing vector store or create new one.
        
        Args:
            documents: List of document chunks to store if creating new
            
        Returns:
            Chroma: Vector store instance
        """
        existing_store = self.load_existing_vectorstore()
        if existing_store is not None:
            print("\nLoading existing vector database...")
            return existing_store
        
        print("\nNo existing database found. Creating new vector database...")
        return self.initialize_vectorstore(documents)
    
    def similarity_search(self, 
                         query: str,
                         k: int = 3) -> List[Document]:
        """
        Perform similarity search on vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Document]: List of relevant documents
        """
        return self.vectorstore.similarity_search(query, k=k)
