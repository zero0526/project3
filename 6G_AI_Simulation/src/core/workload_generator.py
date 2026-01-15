import numpy as np
from typing import List, Dict
from src.entities import Terminal, Task
from src.utils import cfg

class WorkloadGenerator:
    def __init__(
        self, 
        service_config: List[Dict], 
        terminals: List[Terminal],
        workload_config: Dict= cfg.task_param
    ):
        """
        Quản lý việc sinh tải cho toàn mạng.
        Nó chứa các tham số workload toàn cục và danh sách các Terminal.
        
        Args:
            service_config: List chứa thông tin chi tiết các Service (ImageClass, ObjDetect...)
            terminals: Danh sách các đối tượng Terminal đã được khởi tạo
        """
        self.workload_config= workload_config
        self.service_config = service_config
        self.terminals = terminals
        
        # --- Lấy tham số Workload từ cấu hình ---
        self.arrival_rate = self.workload_config.get('arrival_rate', 1.0)
        self.zipf_param = self.workload_config.get("zipf_param", 0.8)
        self.fixed_batch_size = self.workload_config.get("default_batch_size", 20) 

        # Zipf (Pre-calculation)
        self.zipf_probs = self._calculate_zipf_probs(len(self.service_config), self.zipf_param)
        
        print(f"[WorkloadGen] Initialized with {len(terminals)} terminals.")
        print(f"[WorkloadGen] Fixed Batch Size per Task: {self.fixed_batch_size}")
        print(f"[WorkloadGen] Service Probabilities (Zipf s={self.zipf_param}): {self.zipf_probs}")

    def _calculate_zipf_probs(self, n: int, s: float) -> np.ndarray:
        """Tính vector xác suất Zipf cho n dịch vụ."""
        if n == 0: return np.array([])
        ranks = np.arange(1, n + 1)
        weights = 1.0 / np.power(ranks, s)
        probs = weights / np.sum(weights)
        return probs

    def step(self, current_time_slot: int) -> List[Task]:
        """
        Hàm chính được gọi tại mỗi bước mô phỏng (Simulation Step).
        Yêu cầu tất cả các Terminal sinh task.
        
        Returns:
            List các Task được sinh ra trong slot này.
        """
        generated_tasks = []
        
        for terminal in self.terminals:
            # Ủy quyền việc sinh task cho từng Terminal
            # Truyền các tham số workload toàn cục vào cho Terminal xử lý
            task = terminal.step_generate_task(
                current_time_slot=current_time_slot,
                arrival_rate=self.arrival_rate,
                batch_size=self.fixed_batch_size,
                zipf_probs=self.zipf_probs,
                service_config_list=self.service_config
            )
            
            if task:
                generated_tasks.append(task)
                
        return generated_tasks