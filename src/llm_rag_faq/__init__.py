"""Layered modular RAG package for DataTalks.Club course FAQs."""

from __future__ import annotations

from llm_rag_faq.data import load_faq_data
from llm_rag_faq.llm import AnthropicClient, LocalGGUFClient
from llm_rag_faq.rag import RAGBase, RAGResult
from llm_rag_faq.retrieval import HybridFAQSearch, build_index

__all__ = [
    "AnthropicClient",
    "HybridFAQSearch",
    "LocalGGUFClient",
    "RAGBase",
    "RAGResult",
    "build_index",
    "load_faq_data",
]
