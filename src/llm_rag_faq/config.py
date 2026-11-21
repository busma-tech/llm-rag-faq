"""Centralized configuration constants and defaults for ``llm_rag_faq``."""

from __future__ import annotations

# FAQ data ingestion
DOCS_URL = "https://datatalks.club/faq/json/courses.json"
URL_PREFIX = "https://datatalks.club/faq/"
DEFAULT_COURSE = "llm-zoomcamp"
DEFAULT_TIMEOUT = 30.0
DEFAULT_CACHE_DIR = ".cache"

# Hybrid retrieval & embeddings
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BOOST: dict[str, float] = {"question": 3.0, "section": 0.5}
DEFAULT_RRF_K = 60
DEFAULT_MIN_CANDIDATES = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_THREADS = 2

# Local GGUF inference (llama.cpp)
DEFAULT_LOCAL_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_LOCAL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_LOCAL_REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
DEFAULT_N_CTX = 4096
DEFAULT_N_BATCH = 512
DEFAULT_LOCAL_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_SEED = 42
TEMPLATE_OVERHEAD_TOKENS = 64

# Cloud LLM & RAG pipeline
DEFAULT_CLOUD_MODEL = "claude-haiku-5-5"
DEFAULT_CLOUD_MAX_TOKENS = 1024
DEFAULT_CLOUD_NUM_RESULTS = 5
DEFAULT_LOCAL_NUM_RESULTS = 3
DEFAULT_REWRITE_MAX_TOKENS = 64
DEFAULT_MAX_QUERY_CHARS = 2000
NO_ANSWER_RESPONSE = "I don't know."
