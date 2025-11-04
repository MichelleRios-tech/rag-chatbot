import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # LLM Provider settings
    # Auto-detect provider: Anthropic → Gemini → LM Studio (based on API key availability)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Google Gemini settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # LM Studio settings (OpenAI-compatible local model)
    LMSTUDIO_BASE_URL: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    LMSTUDIO_MODEL: str = os.getenv("LMSTUDIO_MODEL", "local-model")
    LMSTUDIO_API_KEY: str = os.getenv("LMSTUDIO_API_KEY", "not-needed")  # LM Studio doesn't require real API key

    # Provider selection (auto, anthropic, gemini, or lmstudio)
    # "auto" will use fallback priority: Anthropic → Gemini → LM Studio
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")

    # Model selection mode (dynamic allows runtime switching, static locks at startup)
    MODEL_SELECTION_MODE: str = os.getenv("MODEL_SELECTION_MODE", "dynamic")

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


