import logging
import os
import sys

# Windows console encoding fix
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def setup_logger(name="6G_Sim", log_file="data/logs/simulation.log", level=logging.INFO, mode='a'):
    """
    Thiết lập cấu hình logging chuẩn cho toàn bộ dự án.
    
    Args:
        name (str): Tên logger (thường là __name__).
        log_file (str): Đường dẫn file log đầu ra.
        level (int): Mức độ log (DEBUG, INFO, WARNING...).
    
    Returns:
        logging.Logger: Đối tượng logger đã cấu hình.
    """
    # 1. Tạo thư mục nếu chưa tồn tại
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 2. Định dạng Log (Format)
    # Ví dụ: 2023-10-25 10:00:00 - 6G_Sim - INFO - Message...
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. Handler 1: Ghi ra File
    file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 4. Handler 2: In ra Console (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 5. Khởi tạo Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Tránh duplicate log nếu setup nhiều lần
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Tạo một instance mặc định để dùng nhanh
# from src.utils.logger import logger
logger = setup_logger()