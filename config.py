# agentic_rag_pipeline/config.py

import os
from dotenv import load_dotenv

# --- Load .env file from the project root ---
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("Warning: .env file not found. Please create one.")

# --- Database Configuration (from Dopa_project) ---
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# --- Model & API Configuration (Consolidated) ---
# LLM for Metadata, Proofreading, etc.
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ptm-gpt-oss-120b")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://3.113.24.61/llm-large-inference/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 120))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))

# Embedding Model
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu") # Change to 'cuda' if you have a GPU

# OCR Service
OCR_API_BASE = os.getenv("OCR_API_BASE", "http://3.113.24.61/typhoon-ocr-service/v1")
OCR_API_KEY = os.getenv("OCR_API_KEY", "not-used")

# --- Pipeline Settings ---
# Default folder to look for new documents
DATA_ROOT_FOLDER = os.getenv("DATA_ROOT_FOLDER", "data/")

# --- Agent Qdrant Configuration ---
AGENT_QDRANT_HOST = os.getenv("AGENT_QDRANT_HOST", "localhost")
AGENT_QDRANT_PORT = int(os.getenv("AGENT_QDRANT_PORT", 6334))
AGENT_QDRANT_COLLECTION_NAME = os.getenv("AGENT_QDRANT_COLLECTION_NAME")