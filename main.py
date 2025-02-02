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

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='PDF RAG Assistant')
    parser.add_argument('--pdf_dirs', type=str, default='documents',
                      help='Directory containing PDF documents')
    parser.add_argument('--db_dir', type=str, default='vector_db',
                      help='Directory for vector database')
    args = parser.parse_args()
    
    try:
        print("Initializing PDF RAG Assistant...")
        
        # Initialize components
        pdf_processor = PDFProcessor()
        vector_manager = VectorStoreManager(persist_directory=args.db_dir)
        metrics = MetricsTracker()
        router = LLMRouter()
        
        print("\n=== Processing Documents ===")
        documents = pdf_processor.process_directory(args.pdf_dirs)
        
        # Get or create vectorstore
        vectorstore = vector_manager.get_or_create_vectorstore(documents)
        
        print("\nDF RAG System Ready!")
        print("Enter your questions about labor and human rights compliance in global supply chains.")
        print("Type 'exit' to quit.\n")
        
        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'exit':
                break
                
            print("\nProcessing your query...")
            
            # Get relevant docs using the manager
            relevant_docs = vectorstore.similarity_search(query, k=3)
            
            # Debug print
            print(f"\nFound {len(relevant_docs)} relevant documents")
            
            result = metrics.track_query(
                "Mistral-7B",
                query,
                lambda: router.generate_response(query, relevant_docs).strip()
            )
            
            print(f"\nAnswer: {result}")
            metrics.print_metrics()
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    exit(main())
