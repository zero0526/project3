import time
from src.utils import cfg  # Giữ import theo yêu cầu cấu trúc dự án

class Task:
    def __init__(
        self, 
        task_id, 
        terminal_id, 
        source_node_id,     # <--- QUAN TRỌNG: ID nút biên nơi task sinh ra (src_i)
        service_id, 
        batch_size: int, 
        deadline: float, 
        min_accuracy: float, 
        created_at: float,
        service_info: dict, 
    ):
        """
        Đại diện cho một yêu cầu tính toán (Inference Request).
        """
        self.id = task_id
        self.terminal_id = terminal_id
        self.source_node_id = source_node_id 
        self.service_id = service_id
        
        self.batch_size = batch_size
        
        # input_data_size: MB per item (Table II)
        unit_size_mb = service_info.get('input_data_size', 0.0) 
        
        self.total_data_size_mb = unit_size_mb * self.batch_size 

        # --- QoS Requirements (SLA) ---
        self.deadline = deadline            # t^{th}_{i,s} (giây)
        self.min_accuracy = min_accuracy    # Acc^{th}_{i,s} (0.0 - 1.0)
        
        # Service Type: 1 (Continuous - Always On), 0 (Occasional - Cold Start)
        self.omega = service_info.get('omega', 1) 

        # --- Thông tin Lập lịch (Sẽ được Agent điền vào sau) ---
        self.assigned_node_id = None    # Nút đích (v) - Kết quả của hành động Offloading
        self.selected_model_idx = None  # Model (b) - Kết quả của hành động Model Selection
        self.required_workload_gflops = 0.0 # Tổng khối lượng tính toán (F)
        
        # --- Kết quả thực thi (Dùng để tính Reward/Log) ---
        self.created_at = created_at
        self.finished_at = None
        
        # Các thành phần độ trễ chi tiết (để debug và vẽ biểu đồ)
        self.transmission_delay = 0.0
        self.queue_delay = 0.0
        self.computation_delay = 0.0
        self.cold_start_delay = 0.0 

    def assign_schedule(self, node_id, model_idx, unit_workload: float):
        """
        Gán quyết định từ Agent (Lower-level) cho Task.
        
        Args:
            node_id: ID nút tính toán được chọn (v).
            model_idx: Index của model được chọn trong danh sách model của service này.
            unit_workload: Khối lượng tính toán (GFLOPS) để xử lý 1 đơn vị dữ liệu (1 item).
        """
        self.assigned_node_id = node_id
        self.selected_model_idx = model_idx
        
        # Công thức (12): Tổng Workload = unit_workload * Batch_size
        self.required_workload_gflops = unit_workload * self.batch_size

    @property
    def total_delay(self):
        """Tổng thời gian xử lý từ lúc gửi đến lúc xong."""
        return (self.transmission_delay + 
                self.queue_delay + 
                self.computation_delay + 
                self.cold_start_delay)

    @property
    def is_successful(self):
        """Kiểm tra Task có hoàn thành trong Deadline không (thỏa mãn SLA)."""
        if self.finished_at is None:
            return False
        # Lưu ý: total_delay là khoảng thời gian trôi qua, so sánh với deadline
        return self.total_delay <= self.deadline

    def __repr__(self):
        status = "DONE" if self.finished_at else "PENDING"
        return (f"<Task {self.id} | Src:{self.source_node_id} | Svc:{self.service_id} | deadline: {self.deadline} "
                f"Data:{self.total_data_size_mb:.2f}MB | Workload:{self.required_workload_gflops:.2f}G | "
                f"Status:{status}>")