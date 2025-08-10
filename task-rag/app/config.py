"""Configuration settings using Pydantic BaseSettings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key for LLM and embeddings")
    model_name: str = Field(default="gpt-4-turbo-preview", description="OpenAI model name for chat completion")
    embeddings_model: str = Field(default="text-embedding-3-small", description="OpenAI embeddings model")
    
    # Telegram Bot Configuration
    telegram_bot_token: str = Field(..., description="Telegram bot token from BotFather")
    telegram_webhook_secret: Optional[str] = Field(default=None, description="Secret token for webhook validation")
    telegram_webhook_url: Optional[str] = Field(default=None, description="Webhook URL for Telegram bot")
    
    # Storage Configuration
    chroma_dir: Path = Field(default=Path("data/chroma"), description="Directory for Chroma vector database")
    uploads_dir: Path = Field(default=Path("data/uploads"), description="Directory for uploaded files")
    
    # Application Configuration
    app_host: str = Field(default="0.0.0.0", description="FastAPI host")
    app_port: int = Field(default=8000, description="FastAPI port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size for document splitting")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks")
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()

