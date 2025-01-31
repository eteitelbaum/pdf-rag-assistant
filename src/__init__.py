"""
Academic RAG System
-----------------
A modular system for analyzing academic papers using RAG.
"""

from .models.llm_router import LLMRouter
from .document_processing.vectorstore import VectorStoreManager
from .utils.document_loader import process_documents
from .utils.metrics import track_resources

__all__ = ['LLMRouter', 'VectorStoreManager', 'process_documents', 'track_resources']
