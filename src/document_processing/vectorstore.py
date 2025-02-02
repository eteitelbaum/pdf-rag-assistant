from typing import List, Optional
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import hashlib
import json
from datetime import datetime

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
    
    def get_or_create_vectorstore(self, documents: Optional[List[Document]] = None):
        if os.path.exists(self.persist_directory):
            print("\nChecking existing vector database...")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            doc_count = vectorstore._collection.count()
            print(f"Found {doc_count} documents in database")
            
            if documents:
                print(f"\nProcessing {len(documents)} documents in batches...")
                batch_size = 32
                total_batches = (len(documents) - 1) // batch_size + 1
                
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

def get_pdf_hash(pdf_path):
    """Generate a hash for a PDF file to track uniqueness"""
    with open(pdf_path, 'rb') as file:
        return hashlib.md5(file.read()).hexdigest()

def load_processed_files():
    """Load record of previously processed files"""
    tracking_file = os.path.join(PERSIST_DIRECTORY, 'processed_files.json')
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files(processed):
    """Save record of processed files"""
    tracking_file = os.path.join(PERSIST_DIRECTORY, 'processed_files.json')
    with open(tracking_file, 'w') as f:
        json.dump(processed, f)

def get_vectorstore(pdf_paths=None):
    try:
        # Initialize or load existing vectorstore
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        if pdf_paths:
            # Load record of processed files
            processed_files = load_processed_files()
            new_pdfs = []
            
            # Check for new PDFs
            for pdf_path in pdf_paths:
                pdf_hash = get_pdf_hash(pdf_path)
                if pdf_hash not in processed_files:
                    new_pdfs.append(pdf_path)
                    processed_files[pdf_hash] = {
                        'path': pdf_path,
                        'processed_date': str(datetime.now())
                    }
            
            # Process only new PDFs
            if new_pdfs:
                print(f"\nFound {len(new_pdfs)} new PDFs to process...")
                documents = load_pdfs(new_pdfs)
                chunks = split_documents(documents)
                print(f"Processing {len(chunks)} new chunks...")
                vectorstore = process_documents_in_batches(chunks, existing_vectorstore=vectorstore)
                
                # Save record of processed files
                save_processed_files(processed_files)
            else:
                print("\nNo new PDFs to process. Using existing embeddings.")
                
        return vectorstore
            
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None