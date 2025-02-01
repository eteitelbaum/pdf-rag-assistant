from typing import List, Optional
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class HuggingFaceEmbeddings:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def embed_documents(self, texts):
        # Tokenize and encode
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]
        
    def embed_query(self, text: str):
        """Embed a single piece of text (the query)"""
        return self.embed_documents([text])[0]

class VectorStoreManager:
    """Manages vector database operations and embeddings"""
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 persist_directory: str = "./academic_db"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to store the vector database
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
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
    
    def similarity_search(self, query: str, k: int = 3):
        """
        Perform similarity search on vector store.
        """
        print("\nDebug: Starting similarity search")
        print(f"Query: {query}")
        print(f"Looking for {k} documents")
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        print("Query embedding created")
        
        # Search
        results = self.collection.search(
            query_embeddings=query_embedding,
            n_results=k
        )
        print(f"Search complete. Found {len(results)} results")
        
        return results