"""
Academic RAG System
-----------------
A modular system for analyzing academic papers using RAG.
"""

from .models.llm_router import LLMRouter
from .document_processing.pdf_loader import PDFProcessor
from .document_processing.vectorstore import VectorStoreManager
from .utils.metrics_tracker import track_resources

__all__ = ['LLMRouter', 'PDFProcessor', 'VectorStoreManager', 'track_resources']
