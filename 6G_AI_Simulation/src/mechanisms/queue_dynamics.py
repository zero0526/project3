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
        remaining = max(0.0, current_backlog - processed_workload)
        next_backlog = remaining + arrival_workload
        
        return next_backlog