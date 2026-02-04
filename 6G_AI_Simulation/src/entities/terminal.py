import numpy as np
import random
from typing import Optional, Dict, List
from .task import Task 

class Terminal:
    def __init__(
        self, 
        terminal_id: str, 
        edge_id: str, 
        arrival_rate: float = 1.0, 
        default_batch_size: int = 10
    ):
        """
        Đại diện cho một Terminal (Agent cấp thấp).
        Nó được gán cố định vào 1 Edge Node khi khởi tạo.
        
        Args:
            terminal_id: ID duy nhất của terminal (VD: "UE_0")
            edge_id: ID của Edge Node mà terminal này kết nối (src_i)
            arrival_rate: Xác suất sinh task trong 1 time slot (0 -> 1.0)
            default_batch_size: Số lượng batch mặc định cho request (Paper: 20)
        """
        self.id = terminal_id
        self.edge_id = edge_id # Nút biên cố định mà Terminal này kết nối (src_i)
        self.arrival_rate = arrival_rate
        self.default_batch_size = default_batch_size
        
        # Trạng thái hiện tại (có thể dùng cho RL sau này)
        self.current_task: Optional[Task] = None
        
        # Thống kê (tùy chọn)
        self.generated_tasks_count = 0
    

    def step_generate_task(
        self, 
        current_time_slot: int, 
        arrival_rate: float,
        batch_size: int, 
        zipf_probs: np.ndarray, 
        service_config_list: List[Dict]
    ) -> Optional[Task]:
        """
        Hàm được WorkloadGenerator gọi tại mỗi time slot.
        Cố gắng sinh ra một Task cho time slot hiện tại.
        
        Args:
            current_time_slot: Thời điểm hiện tại (tau)
            arrival_rate: Xác suất sinh task cho terminal này trong slot này.
            batch_size: Số lượng batch cố định cho mỗi task.
            zipf_probs: Mảng xác suất Zipf đã tính sẵn (từ Generator).
            service_config_list: Danh sách cấu hình toàn bộ dịch vụ (từ Generator).
        """
        # if np.random.random() > arrival_rate:
        #     self.current_task = None
        #     return None

        # 2. Chọn 1 Service duy nhất dựa trên phân phối Zipf
        num_services = len(service_config_list)
        selected_service_id_index = np.random.choice(
            np.arange(num_services), 
            p=zipf_probs
        )       
        svc_info = next((s for s in service_config_list if s.get('id') == selected_service_id_index), None) 
        task_acc= [m.get('accuracy', 0.0) for m in svc_info['models']]
        mu = np.mean(task_acc) 
        sigma = 0.1
        min_acc_required = min(np.random.normal(mu, sigma), max(task_acc))
        # 4. Tạo đối tượng Task
        task_id = f"T_{self.id}_{current_time_slot}"
        deadline = random.gauss(svc_info.get('mean_deadline'), svc_info.get('std_deadline'))
        new_task = Task(
            task_id=task_id,
            terminal_id=self.id,
            source_node_id=self.edge_id, 
            service_id=selected_service_id_index, 
            batch_size=batch_size,
            deadline=max(deadline, 1.0),
            min_accuracy=min_acc_required,   
            created_at=current_time_slot,
            service_info=svc_info 
        )
        
        self.current_task = new_task
        self.generated_tasks_count += 1
        
        return new_task

    def __repr__(self):
        return f"<Terminal {self.id} @ {self.edge_id}>"