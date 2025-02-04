from typing import List, Optional
from tqdm import tqdm
from langchain.schema import Document
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import hashlib
import json
from datetime import datetime

# Configuration for persistence - will use environment variable if set, otherwise default
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', './academic_db')
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

class HuggingFaceEmbeddings:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Explicitly set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
    
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
                 persist_directory: str = PERSIST_DIRECTORY):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to store the vector database
        """
        print("\n=== VectorStoreManager Initialization ===")
        print(f"Module location: {__file__}")
        print(f"Embedding model: {embedding_model}")
        print(f"Persist directory: {persist_directory}")
        
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.persist_directory = persist_directory
        
        # Initialize tracking file path
        self.tracking_file = os.path.join(persist_directory, 'processed_files.json')
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        print("VectorStoreManager initialization complete")
    
    def get_pdf_hash(self, pdf_path: str) -> str:
        """Generate a hash for a PDF file to track uniqueness"""
        with open(pdf_path, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()
    
    def load_processed_files(self) -> dict:
        """Load record of previously processed files"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_processed_files(self, processed: dict):
        """Save record of processed files"""
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(processed, f)
    
    def track_processed_file(self, pdf_path: str):
        """Track a newly processed PDF file"""
        processed_files = self.load_processed_files()
        pdf_hash = self.get_pdf_hash(pdf_path)
        processed_files[pdf_hash] = {
            'path': pdf_path,
            'processed_date': str(datetime.now())
        }
        self.save_processed_files(processed_files)
    
    def initialize_vectorstore(self, documents: List[Document]) -> Chroma:
        """Initialize or load vector store with documents."""
        print("\nGenerating embeddings and storing in database...")
        try:
            with tqdm(total=len(documents), desc="Generating embeddings", unit="chunk") as pbar:
                def progress_callback(current, total):
                    pbar.update(1)
                
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                    progress_callback=progress_callback
                )
            print("\nVector database successfully created and stored!")
            return vectorstore
        except Exception as e:
            print(f"\nError creating vector database: {e}")
            raise
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store if available."""
        try:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except Exception:
            return None
    
    def get_or_create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Get existing vectorstore or create new one if it doesn't exist."""
        return self.process_documents(documents)
    
    def process_documents(self, documents: List[Document]) -> Chroma:
        """Process documents and add to vector store."""
        if os.path.exists(self.persist_directory):
            print("\nFound existing vector database")
            
            if not documents:
                print("No new documents to process. Using existing database.")
                return self.load_existing_vectorstore()
            
            print(f"Found {len(documents)} documents to process")
            vectorstore = self.load_existing_vectorstore()
            
            if documents:
                batch_size = 64
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    try:
                        print(f"\nProcessing batch {i//batch_size + 1}/{total_batches}")
                        vectorstore.add_texts([doc.page_content for doc in batch])
                        print(f"Current document count: {vectorstore._collection.count()}")
                    except Exception as e:
                        print(f"Error in batch {i//batch_size + 1}: {e}")
                
                final_count = vectorstore._collection.count()
                print(f"\nFinished processing. Total documents: {final_count}")
            
            return vectorstore
        
        print("\nNo existing database found. Creating new vector database...")
        return self.initialize_vectorstore(documents)
    
    def similarity_search(self, query: str, k: int = 3):
        """Perform similarity search on vector store."""
        print("\nDebug: Starting similarity search")
        try:
            vectorstore = self.load_existing_vectorstore()
            
            results = vectorstore.similarity_search(query, k=k)
            print(f"\nDebug: Found {len(results)} results")
            
            # Debug information about results
            for i, doc in enumerate(results):
                print(f"\nDocument {i} type: {type(doc)}")
                print(f"Document {i} attributes: {dir(doc)}")
                print(f"Document {i} metadata: {getattr(doc, 'metadata', 'No metadata')}")
            
            # Extract text content directly
            formatted_results = []
            for doc in results:
                try:
                    if hasattr(doc, 'page_content'):
                        formatted_results.append(doc.page_content)
                        print(f"\nSuccessfully extracted content from document")
                    else:
                        print(f"\nDocument missing page_content: {type(doc)}")
                except Exception as e:
                    print(f"\nError processing document: {e}")
            
            if not formatted_results:
                print("\nWarning: No valid documents found in search results")
                
            return formatted_results
            
        except Exception as e:
            print(f"\nError in similarity search: {e}")
            return []