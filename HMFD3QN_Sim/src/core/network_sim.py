import numpy as np
from collections import deque
from src.utils.math_ops import MathOps

class ComputingNode:
    def __init__(self, node_id, specs, config):
        """
        specs: {'cpu': 5600, 'ram': 20, 'hdd': 80}
        """
        self.id = node_id
        self.specs = specs
        self.config = config
        
        # Trạng thái tài nguyên
        self.used_ram = 0
        self.used_hdd = 0
        
        # Service Placement State: {service_id: bool}
        self.placed_services = {} 
        
        # Hàng đợi tác vụ: {service_id: deque([Task objects...])}
        self.task_queues = {} 
        # Backlog ảo (GFLOPs) dùng cho tính toán KKT/RL
        self.queue_backlogs = {} 
        
        # Tham số cho Cold Start
        self.service_status = {} # 'warm' or 'cold'
        self.service_last_active = {}

    def deploy_services(self, placement_action, service_profiles):
        """
        Hành động của Upper Agent đầu mỗi Time Frame.
        placement_action: List [0, 1, 0...] biểu thị đặt service nào.
        """
        # Reset tài nguyên
        self.used_ram = 0
        self.used_hdd = 0
        self.placed_services = {}
        
        for svc_id, place in enumerate(placement_action):
            if place == 1:
                profile = service_profiles[svc_id]
                # Kiểm tra ràng buộc tài nguyên (Constraint Eq. 4 & 5)
                if profile['type'] == 'continuous':
                    if self.used_ram + profile['size'] > self.specs['ram']: continue
                    self.used_ram += profile['size']
                else: # occasional
                    if self.used_hdd + profile['size'] > self.specs['hdd']: continue
                    self.used_hdd += profile['size']
                
                # Chấp nhận đặt dịch vụ
                self.placed_services[svc_id] = True
                if svc_id not in self.task_queues:
                    self.task_queues[svc_id] = deque()
                    self.queue_backlogs[svc_id] = 0.0
                    self.service_status[svc_id] = 'cold' if profile['type'] == 'occasional' else 'warm'

    def admit_task(self, task):
        """
        Nhận task từ Terminal gửi đến.
        """
        if task.service_id in self.placed_services:
            self.task_queues[task.service_id].append(task)
            # Cập nhật backlog (GFLOPS cần xử lý)
            # Giả sử task object có thuộc tính 'gflops'
            self.queue_backlogs[task.service_id] += task.gflops
            return True
        else:
            # Task bị drop vì node không chứa dịch vụ này
            return False

    def process_one_timeslot(self, time_slot_duration, service_profiles):
        """
        Hàm vật lý chạy mỗi Time Slot.
        1. Tính toán KKT để chia CPU.
        2. Xử lý Task trong hàng đợi.
        3. Trả về kết quả QoS và Năng lượng.
        """
        active_services = list(self.placed_services.keys())
        if not active_services:
            return [], 0.0 # Không làm gì

        # 1. Chuẩn bị tham số cho KKT
        # Ở đây ta giả lập G(tau) và Z(tau) đơn giản tỷ lệ với Queue Backlog
        # Trong thực tế cần công thức (24) chính xác nếu train RL
        E_params = {
            'G': [self.queue_backlogs[s] for s in active_services], 
            'Z': [1.0 for _ in active_services] # Z thường là hằng số liên quan đến hệ số năng lượng
        }
        
        # 2. Giải bài toán phân bổ tài nguyên
        f_allocs = MathOps.solve_resource_allocation_kkt(
            Q_vals=[self.queue_backlogs[s] for s in active_services],
            E_params=E_params,
            f_max=self.specs['cpu'],
            num_services=len(active_services)
        )
        
        total_energy = 0.0
        completed_tasks = []

        # 3. Thực hiện xử lý cho từng dịch vụ
        for idx, svc_id in enumerate(active_services):
            f_val = f_allocs[idx] # GFLOPS cấp cho dịch vụ này
            if f_val <= 0: continue
            
            # Khả năng xử lý trong slot này (GFLOPS)
            capacity = f_val * time_slot_duration
            
            # Tính năng lượng (Eq. 14)
            # Lấy đại diện 1 task để tính complexity
            sample_complexity = service_profiles[svc_id]['gflops'] 
            is_occ = (service_profiles[svc_id]['type'] == 'occasional')
            
            # Kiểm tra Cold Start
            cold_delay = 0
            cold_energy = 0
            if is_occ and self.service_status[svc_id] == 'cold' and self.queue_backlogs[svc_id] > 0:
                self.service_status[svc_id] = 'warm'
                cold_delay = self.config['cold_start_delay']
                cold_energy = self.config['cold_start_energy']
            
            # Năng lượng tính toán động
            # E ~ capacity_used * f^2
            # Đơn giản hóa: E = Power * Time = (coeff * f^3) * time_slot
            e_dynamic = self.config['coeff_energy'] * (f_val ** 3) * time_slot_duration
            total_energy += (e_dynamic + cold_energy)

            # 4. Giảm hàng đợi (Queue Dynamics)
            # Trừ backlog trước
            processed_amount = min(self.queue_backlogs[svc_id], capacity)
            self.queue_backlogs[svc_id] -= processed_amount
            
            # Xử lý các task cụ thể trong deque (FIFO)
            # Capacity còn lại để trừ vào size của từng task
            remaining_cap = processed_amount
            
            while self.task_queues[svc_id] and remaining_cap > 0:
                task = self.task_queues[svc_id][0] # Xem task đầu
                
                # Nếu task chưa có thuộc tính 'remaining_gflops', gán ban đầu
                if not hasattr(task, 'remaining_gflops'):
                    task.remaining_gflops = task.gflops
                
                if task.remaining_gflops <= remaining_cap:
                    # Task hoàn thành
                    remaining_cap -= task.remaining_gflops
                    finished_task = self.task_queues[svc_id].popleft()
                    
                    # Tính toán QoS
                    processing_time = (task.gflops / f_val) + cold_delay # Ước lượng
                    # Lưu ý: Thời gian thực tế là (Current Time - Arrival Time)
                    # Ta sẽ tính delay tổng ở lớp Environment
                    completed_tasks.append(finished_task)
                else:
                    # Task chưa xong, trừ đi phần đã làm
                    task.remaining_gflops -= remaining_cap
                    remaining_cap = 0
                    
        return completed_tasks, total_energy

    def get_observation(self):
        """
        Trả về vector trạng thái cho RL (Queue lengths, Resource usage...)
        """
        # Vector hóa trạng thái cho Upper Agent
        return {
            'queues': list(self.queue_backlogs.values()),
            'cpu_util': sum(self.queue_backlogs.values()) / self.specs['cpu'] # Ước lượng
        }