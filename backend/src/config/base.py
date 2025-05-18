import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration class"""

    # Application settings
    APP_NAME: str = "ML/RL Backend"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ML settings
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and not callable(getattr(cls, key))
        }