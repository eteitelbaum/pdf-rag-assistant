"""
Academic RAG System
-----------------
A modular system for analyzing academic papers using RAG.
"""

from .models import LLMRouter, AcademicDatabaseConnector
from .document_processing import PDFProcessor, VectorStoreManager
from .utils import MetricsTracker

__all__ = [
    'LLMRouter',
    'AcademicDatabaseConnector',
    'PDFProcessor',
    'VectorStoreManager',
    'MetricsTracker'
]
