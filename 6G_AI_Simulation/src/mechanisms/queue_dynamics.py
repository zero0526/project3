import numpy as np

class QueueDynamics:
    """
    Thực hiện logic cập nhật hàng đợi theo lý thuyết Lyapunov (Eq. 16).
    """
    
    @staticmethod
    def update_backlog(current_backlog, processed_workload, arrival_workload):
        """
        Tính Q(t+1) dựa trên Q(t), W(t) và A(t).
        
        Args:
            current_backlog (float): Q(t) - Backlog hiện tại (GFLOPS).
            processed_workload (float): W(t) - Khối lượng đã xử lý (f * duration).
            arrival_workload (float): A(t) - Khối lượng task mới nhận vào.
            
        Returns:
            float: Q(t+1) - Backlog tiếp theo.
        """
        # Công thức: Q_next = max(Q_curr - W, 0) + A
        
        # 1. Trừ đi phần đã xử lý (không được âm)
        remaining = max(0.0, current_backlog - processed_workload)
        
        # 2. Cộng thêm phần mới đến
        next_backlog = remaining + arrival_workload
        
        return next_backlog

    @staticmethod
    def calculate_virtual_queue(backlog, epsilon=1e-5):
        """
        (Tùy chọn) Tính hàng đợi ảo nếu có ràng buộc độ trễ khắt khe.
        Trong bài báo gốc dùng Backlog thực tế, nhưng hàm này có thể mở rộng
        cho các thuật toán nâng cao hơn.
        """
        return backlog