class Task:
        def __init__(self, task_id, service_id, terminal_id, unit_size, batch_size, deadline, min_accuracy, omega, created_at):
            self.id = task_id
            self.service_id = service_id
            self.terminal_id = terminal_id
            
            # --- Yêu cầu từ người dùng (User Requirements) ---
            self.unit_size = unit_size      # Kích thước 1 unit (MB)
            self.batch_size = batch_size    # Số lượng unit (Batch size)
            self.data_size = self.unit_size * self.batch_size 

            self.deadline = deadline      # t^{th}_{i,s} (s)
            self.min_accuracy = min_accuracy # Acc^{th}_{i,s} (New!)
            self.omega = omega            # Loại dịch vụ
            
            # --- Thông tin xử lý (Sẽ được điền sau khi Agent chọn Model) ---
            self.selected_model_idx = None  # Model ID (b)
            self.required_workload = 0.0   # F(m_{s,b}) - GFLOPS
            
            # --- Tem thời gian ---
            self.created_at = created_at
            self.arrival_time_at_node = None

        def __repr__(self):
            return f"<Task {self.id} | Svc:{self.service_id} | MinAcc:{self.min_accuracy:.2f}>"