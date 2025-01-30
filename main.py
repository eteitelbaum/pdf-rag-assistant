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

def main():
    """Main execution function"""
    print("Initializing PDF RAG Assistant...")
    
    # Initialize components first
    pdf_processor = PDFProcessor()
    vector_manager = VectorStoreManager()
    metrics = MetricsTracker()
    
    print("\n=== Step 1: Processing Documents ===")
    print("\nProcessing PDF documents...")
    documents = pdf_processor.process_directory("documents")
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
    
    # Show initial system resources
    initial_resources = metrics.track_system_resources()
    print("\nSystem Resources:")
    print(f"Available Memory: {initial_resources['memory_available_gb']:.1f} GB")
    print(f"CPU Cores: {initial_resources['cpu_count']}")
    
    # Show resource usage after processing
    post_process_resources = metrics.track_system_resources()
    print("\nResource Usage After Processing:")
    print(f"Memory Used: {post_process_resources['memory_used_mb']:.1f} MB")
    print(f"Memory Usage: {post_process_resources['memory_percent']:.1f}%")
    print(f"CPU Usage: {post_process_resources['cpu_percent']:.1f}%")
    
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
        
        print("\nPerformance Metrics:")
        print(f"Time taken: {result['duration_seconds']:.2f} seconds")
        print(f"Memory used: {resources['memory_used_mb']:.1f} MB")
        print(f"Memory usage: {resources['memory_percent']:.1f}%")
        print(f"CPU usage: {resources['cpu_percent']:.1f}%")
        print("-" * 80)
        
    return 0

if __name__ == "__main__":
    exit(main())
