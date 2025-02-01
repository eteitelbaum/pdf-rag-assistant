from typing import List
import os
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

class PDFProcessor:
    """Handles PDF document loading and processing"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdfs(self, pdf_directory: str) -> List[Document]:
        """
        Load and process PDF documents with progress tracking.
        
        Args:
            pdf_directory: Path to directory containing PDFs
            
        Returns:
            List[Document]: List of processed document chunks
        """
        documents = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_directory}")
        
        print("\nLoading PDFs:")
        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            file_path = os.path.join(pdf_directory, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Add source filename to metadata
                for doc in docs:
                    doc.metadata['source'] = filename
                documents.extend(docs)
            except Exception as e:
                print(f"\nError loading {filename}: {e}", file=sys.stderr)
                continue
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        print(f"\nSplitting {len(documents)} documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        return splits
    
    def process_directory(self, pdf_directory: str) -> List[Document]:
        """
        Process all PDFs in a directory - load and split into chunks.
        
        Args:
            pdf_directory: Path to directory containing PDFs
            
        Returns:
            List[Document]: List of processed and split document chunks
        """
        # First load the PDFs
        documents = self.load_pdfs(pdf_directory)
        
        # Then split them into chunks
        return self.split_documents(documents)
