import numpy as np

class Task:
    def __init__(self, task_id, service_id, terminal_id, size, workload, deadline, min_acc, created_at):
        self.id = task_id
        self.service_id = service_id
        self.terminal_id = terminal_id
        self.size = size          # Kích thước dữ liệu (MB)
        self.workload = workload  # Khối lượng tính toán (GFLOPs)
        self.deadline = deadline  # SLA deadline (s)
        self.min_acc = min_acc    # Độ chính xác yêu cầu
        self.created_at = created_at
    
    def __repr__(self):
        return f"<Task {self.id} | Svc: {self.service_id} | Size: {self.size:.2f}MB | Load: {self.workload:.2f}G | Term: {self.terminal_id}>"

class WorkloadGenerator:
    def __init__(self, config):
        self.num_services = config.get('num_services', 5)
        self.zipf_param = config.get('zipf_param', 0.8) # Bài báo dùng a=0.8
        self.arrival_rate = config.get('arrival_rate', 20) # Lambda: tasks/slot
        self.terminals = config.get('terminals', []) # Danh sách ID terminal
        
        # Cấu hình chi tiết cho 5 loại dịch vụ (Theo bài báo hoặc giả lập)
        # [Size(MB), GFLOPs, Deadline(s), MinAcc]
        self.service_profiles = {
            0: {'size': 2, 'gflops': 1.5, 'deadline': 0.5, 'acc': 0.9, 'type': 'continuous'}, # Image Class
            1: {'size': 3, 'gflops': 2.0, 'deadline': 0.6, 'acc': 0.85, 'type': 'continuous'}, 
            2: {'size': 4, 'gflops': 4.5, 'deadline': 0.8, 'acc': 0.9, 'type': 'occasional'}, # Object Detect
            3: {'size': 5, 'gflops': 5.0, 'deadline': 1.0, 'acc': 0.8, 'type': 'occasional'}, 
            4: {'size': 10, 'gflops': 8.0, 'deadline': 1.5, 'acc': 0.95, 'type': 'continuous'}, # Video
        }
        
        # Pre-calculate Zipf probabilities for efficiency
        # Zipf sinh ra 1, 2, 3... ta cần map về 0, 1, 2...
        x = np.arange(1, self.num_services + 1)
        weights = x ** (-self.zipf_param)
        self.service_probs = weights / weights.sum()

    def generate(self, time_slot):
        """
        Sinh ra danh sách các Task cho một Time Slot cụ thể.
        """
        generated_tasks = []
        
        # 1. Xác định số lượng task đến trong slot này (Phân phối Poisson)
        # Nếu mô phỏng mỗi terminal sinh tối đa 1 task/slot như bài báo:
        # num_tasks = len(self.terminals) 
        # Hoặc dùng Poisson cho toàn mạng:
        num_tasks = np.random.poisson(self.arrival_rate)
        
        # 2. Sinh task chi tiết
        for _ in range(num_tasks):
            # Chọn ngẫu nhiên Terminal gửi yêu cầu
            term_id = np.random.choice(self.terminals)
            
            # Chọn loại dịch vụ theo phân phối Zipf
            svc_id = np.random.choice(np.arange(self.num_services), p=self.service_probs)
            
            profile = self.service_profiles[svc_id]
            
            # Tạo đối tượng Task
            # Tạo đối tượng Task
            # Thêm tính ngẫu nhiên vào Size hoặc GFLOPs: +/- 10%
            variation_factor = np.random.uniform(0.9, 1.1)
            actual_size = profile['size'] * variation_factor
            actual_gflops = profile['gflops'] * variation_factor

            task = Task(
                task_id=f"{time_slot}_{term_id}_{np.random.randint(1000)}",
                service_id=svc_id,
                terminal_id=term_id,
                size=actual_size,
                workload=actual_gflops,
                deadline=profile['deadline'],
                min_acc=profile['acc'],
                created_at=time_slot
            )
            generated_tasks.append(task)
            
        return generated_tasks

# --- Test Script ---
if __name__ == "__main__":
    conf = {
        'num_services': 5,
        'zipf_param': 0.8,
        'arrival_rate': 10,
        'terminals': [f"Term_{i}" for i in range(20)]
    }
    gen = WorkloadGenerator(conf)
    
    # Test sinh task cho slot 0
    tasks = gen.generate(time_slot=0)
    print(f"Generated {len(tasks)} tasks.")
    
    # Kiểm tra phân phối Zipf (chạy nhiều lần)
    svc_counts = {i:0 for i in range(5)}
    for _ in range(1000):
        ts = gen.generate(0)
        for t in ts:
            svc_counts[t.service_id] += 1
    print("Service distribution (Should follow Zipf - Svc 0 highest):")
    print(svc_counts)