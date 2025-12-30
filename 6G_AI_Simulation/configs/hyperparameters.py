from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Tuple
import os

def get_env_file() -> str:
    return os.environ.get("ENV_FILE", "dev.env")

class Hyperparams(BaseSettings):
    # 1. Training Loop Control
    SEED: int = Field(default=42)
    TOTAL_EPISODES: int = Field(default=50)
    STEPS_PER_EPISODE: int = Field(default=100)
    SLOT_DURATION: float = Field(default=0.1)
    SLOTS_PER_FRAME: int = Field(default=10)
    
    # 2. Workload Parameters
    ARRIVAL_RATE: float = Field(default=30.0)
    ZIPF_PARAM: float = Field(default=0.8)
    
    # 3. Upper Agent (Service Placement)
    UPPER_LR: float = Field(default=1e-4)
    UPPER_BATCH_SIZE: int = Field(default=32)
    UPPER_GAMMA: float = Field(default=0.99)
    UPPER_TAU: float = Field(default=0.005)
    UPPER_BUFFER_CAPACITY: int = Field(default=10000)
    UPPER_TEMP: float = Field(default=1.0)
    
    # 4. Lower Agent (Offloading)
    LOWER_LR: float = Field(default=1e-4)
    LOWER_BATCH_SIZE: int = Field(default=32)
    LOWER_GAMMA: float = Field(default=0.99)
    LOWER_TAU: float = Field(default=0.005)
    LOWER_BUFFER_CAPACITY: int = Field(default=10000)
    LOWER_TEMP: float = Field(default=1.0)
    
    # 5. Network Parameters
    DEFAULT_BANDWIDTH: float = Field(default=600.0)
    PROPAGATION_DELAY: float = Field(default=0.005)
    TRANSMISSION_POWER: float = Field(default=0.2)

    # 6. Paths & Logging
    TOPOLOGY_XML: str = Field(default="configs/topologies/atlanta.xml")
    TOPOLOGY_JSON: str = Field(default="configs/topologies/atlanta_processed.json")
    LOGS_DIR: str = Field(default="data/logs/")
    CHECKPOINT_DIR: str = Field(default="data/models/")

    # 7. Algorithm Specific Params
    EPS: float = Field(default=1e-9)
    MIN_BATCH_SIZE: int = Field(default=1)
    MAX_BATCH_SIZE: int = Field(default=5)
    V: float = Field(default=1e-7)
    EPS_COLD_E: float = Field(default=0.2)
    EPS_COLD_T: Tuple[float, float] = Field(default=(0.15, 0.85))

    # Cấu hình Pydantic v2
    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_file_encoding="utf-8",
        # Cho phép các biến môi trường viết thường khớp với biến khai báo viết hoa
        case_sensitive=False, 
        # Quan trọng: Không báo lỗi nếu file .env có dư biến
        extra="ignore" 
    )

# Khởi tạo instance để sử dụng trong toàn dự án
hp = Hyperparams()