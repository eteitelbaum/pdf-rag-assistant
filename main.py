"""
PDF RAG Assistant
------------------------

Main entry point for the RAG system that analyzes documents
and provides intelligent responses based on their content.

Usage:
    python main.py

The system will:
1. Load PDFs from the 'documents' directory
2. Process and store document embeddings
3. Use Mistral-7B for answering queries
4. Track performance metrics
"""

import os
import psutil
import argparse
from src import (
    LLMRouter,
    PDFProcessor,
    VectorStoreManager,
    MetricsTracker
)

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("documents", exist_ok=True)
    os.makedirs("vector_db", exist_ok=True)

def create_vectorstore(documents, db_dir):
    try:
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        print(f"\nCreating vector database in: {db_dir}")
        vectorstore = VectorStore(documents, db_dir)
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        return None

def main():
    """Main execution function"""
    # Add argument parsing
    parser = argparse.ArgumentParser(description='PDF RAG Assistant')
    parser.add_argument('--pdf_dirs', type=str, default='documents',
                      help='Directory containing PDF documents')
    parser.add_argument('--db_dir', type=str, default='vector_db',
                      help='Directory for vector database')
    args = parser.parse_args()
    
    try:
        print("Initializing PDF RAG Assistant...")
        
        # Initialize components first
        pdf_processor = PDFProcessor()
        vector_manager = VectorStoreManager()
        metrics = MetricsTracker()
        
        print("\n=== Step 1: Processing Documents ===")
        print("\nProcessing PDF documents...")
        documents = pdf_processor.process_directory(args.pdf_dirs) 
        vectorstore = vector_manager.get_or_create_vectorstore(documents)
        print("Document processing complete!")
        
        print("\n=== Step 2: Loading Language Model ===")
        # 2. Then initialize the LLM
        router = LLMRouter()
        
        print("\n=== Step 3: Ready for Queries ===")
        
        # 3. After this, queries will be fast because:
        #    - Documents are already processed
        #    - Embeddings are stored in vectorstore
        #    - Model is loaded in memory
        
        # Resource tracking with error handling
        resources = {
            'memory_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
        }

        print("\nSystem Resources:")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"CPU Cores: {psutil.cpu_count()}")

        print("\nResource Usage After Processing:")
        print(f"Memory Used: {resources.get('memory_used_mb', 0):.1f} MB")
        print(f"Memory Usage: {resources.get('memory_percent', 0):.1f}%")
        print(f"CPU Usage: {resources.get('cpu_percent', 0):.1f}%")
        
        # Interactive query loop
        print("\nPDF RAG System Ready!")
        print("Enter your questions about labor and human rights compliance in global supply chains.")
        print("Type 'exit' to quit.\n")
        
        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'exit':
                break
            
            print("\nProcessing your query...\n")
            
            # Get relevant documents
            relevant_docs = vector_manager.similarity_search(vectorstore, query)
            
            # Generate response
            result = metrics.track_query(
                "Mistral-7B",
                query,
                lambda: router.generate_response(query, relevant_docs).strip()
            )
            
            # Track resources
            resources = metrics.track_system_resources()
            
            # Display results
            if 'response' in result:
                print("\nAnswer:", result['response'])
            else:
                print("\nError:", result.get('error', 'Unknown error occurred'))
            
            print_metrics(result, resources)
        
        return 0

    except Exception as e:
        print(f"Error tracking resources: {str(e)}")

def print_metrics(result, resources):
    try:
        print("\nPerformance Metrics:")
        print(f"Time taken: {result.get('duration_seconds', 0):.2f} seconds")
        print(f"Memory used: {float(resources.get('memory_used_mb', 0)):.1f} MB")
        print(f"Memory usage: {float(resources.get('memory_percent', 0)):.1f}%")
        print(f"CPU usage: {float(resources.get('cpu_percent', 0)):.1f}%")
    except Exception as e:
        print(f"Error displaying metrics: {str(e)}")

if __name__ == "__main__":
    exit(main())
