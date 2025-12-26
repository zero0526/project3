# Core Environment

Trái tim của hệ thống mô phỏng.

- **`environment.py`**: Lớp `HMFD3QNEnv` kế thừa `gymnasium.Env`. Quản lý vòng lặp `step()`, `reset()`, và tính toán Reward.
- **`workload_generator.py`**: Sinh các Task ngẫu nhiên theo phân phối Poisson (thời điểm đến) và Zipf (độ phổ biến dịch vụ).
- **`time_manager.py`**: Quản lý chuyển đổi giữa Time Slot (cho Terminal) và Time Frame (cho Node).