import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Cố định hạt giống ngẫu nhiên (Random Seed) cho tất cả các thư viện.
    Giúp kết quả mô phỏng có thể tái lập (Reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Đảm bảo tính tất định (Deterministic) cho CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random Seed set to: {seed}")

def min_max_normalize(value, min_val, max_val):
    """
    Chuẩn hóa giá trị về khoảng [0, 1].
    Dùng cho State Space của RL Agent.
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def safe_divide(numerator, denominator, epsilon=1e-8):
    """Chia an toàn tránh lỗi chia cho 0"""
    return numerator / (denominator + epsilon)

def moving_average(data, window_size=10):
    """
    Làm mượt biểu đồ Reward/Loss.
    Dùng khi vẽ biểu đồ báo cáo.
    """
    if len(data) < window_size:
        return np.mean(data) if len(data) > 0 else 0
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def softmax_stable(x):
    """
    Hàm Softmax ổn định số học (tránh NaN khi exp quá lớn).
    Input: numpy array 1D
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()