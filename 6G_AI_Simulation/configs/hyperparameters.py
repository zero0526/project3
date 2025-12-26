from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Dict
import os
import base64

def get_env_file() -> str:
    if "ENV_FILE" in os.environ:
        return os.environ["ENV_FILE"]
    return "dev.env"

class Hyperparams(BaseSettings):
    
    # Training Loop Control
    SEED: int = Field(default=42)
    TOTAL_EPISODES: int = Field(default=50)
    STEPS_PER_EPISODE: int = Field(default=100)
    
    # Upper Agent (Placement)
    UPPER_LR: float = Field(default=1e-4)
    UPPER_BATCH_SIZE: int = Field(default=32)
    UPPER_GAMMA: float = Field(default=0.99)
    UPPER_TAU: float = Field(default=0.005)
    UPPER_BUFFER_CAPACITY: int = Field(default=10000)
    UPPER_TEMP: float = Field(default=1.0)
    
    # Lower Agent (Offloading)
    LOWER_LR: float = Field(default=1e-4)
    LOWER_BATCH_SIZE: int = Field(default=32)
    LOWER_GAMMA: float = Field(default=0.99)
    LOWER_TAU: float = Field(default=0.005)
    LOWER_BUFFER_CAPACITY: int = Field(default=10000)
    LOWER_TEMP: float = Field(default=1.0)
    
    # Paths & Logging
    LOGS_DIR: str = Field(default="data/logs/")
    CHECKPOINT_DIR: str = Field(default="data/models/")

    EPS: float = Field(default= 1e-9)
    # batch_size for task segment
    MIN_BATCH_SIZE: int= Field(default=1)
    MAX_BATCH_SIZE: int= Field(default=5)
    V: float= Field(default= 1e-7)
    EPS_COLD_E: float= Field(default= 0.2)
    EPS_COLD_T: tuple[float]= Field(default=(0.15, 0.85))
    class Config:
        env_file = get_env_file()
        env_file_encoding = "utf-8"


