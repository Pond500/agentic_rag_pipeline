# agentic_rag_pipeline/core/llm_provider.py

from llama_index.llms.openai_like import OpenAILike
from sentence_transformers import SentenceTransformer

# Import our central config
from agentic_rag_pipeline import config

# --- Global cache for models to avoid reloading ---
_llm_instance = None
_embed_model_instance = None

def get_llm():
    """
    Provides a singleton instance of the Language Model (LLM).
    Loads the model on first call and returns the cached instance subsequently.
    """
    global _llm_instance
    if _llm_instance is None:
        print("Initializing LLM for the first time...")
        _llm_instance = OpenAILike(
            model=config.LLM_MODEL_NAME,
            api_base=config.LLM_API_BASE,
            api_key=config.LLM_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            is_chat_model=True,
            timeout=config.LLM_TIMEOUT,
        )
        print("LLM Initialized.")
    return _llm_instance

def get_embed_model():
    """
    Provides a singleton instance of the Embedding Model.
    Loads the model on first call and returns the cached instance subsequently.
    """
    global _embed_model_instance
    if _embed_model_instance is None:
        print(f"Loading Embedding Model ({config.EMBED_MODEL_NAME}) for the first time...")
        _embed_model_instance = SentenceTransformer(
            config.EMBED_MODEL_NAME,
            device=config.EMBED_DEVICE
        )
        print("Embedding Model Loaded.")
    return _embed_model_instance