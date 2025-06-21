"""
Configuration settings for the ChronoLedge application.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    OPENAI_API_KEY: str = ""
    WIKIMEDIA_API_KEY: Optional[str] = None
    SCRAPEAPI_KEY: str = ""
        
    # Application
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # ETL Settings
    ETL_BATCH_SIZE: int = 100
    ETL_UPDATE_INTERVAL: int = 3600  # seconds
    
    # Reasoning Module Configuration
    REASONING_PROVIDER: str = "openai"  # "openai" or "claude"
    
    # OpenAI Configuration
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.3
    
    # Claude Configuration (placeholder)
    CLAUDE_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-sonnet-20240229"
    CLAUDE_MAX_TOKENS: int = 4000
    CLAUDE_TEMPERATURE: float = 0.3
    
    # Legacy settings for backward compatibility
    LLM_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True) 
