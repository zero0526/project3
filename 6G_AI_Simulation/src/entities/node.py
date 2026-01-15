from collections import deque, defaultdict
import numpy as np
import random
from src.mechanisms.kkt_solver import KKTSolver
from src.mechanisms.energy_model import NodeModel
from src.mechanisms.queue_dynamics import QueueDynamics
from src.entities.task import Task
from src.utils import cfg

class ComputingNode:
    def __init__(self, node_id, specs):
        self.id = node_id
        self.specs = specs
        self.cpu_capacity = specs['cpu']      # f_v(tau)
        self.ram_capacity = specs['ram']
        self.hdd_capacity = specs['hdd']
        self.energy_coeff = specs['energy_coeff'] # epsilon_c
        self.type= specs['type']
        
        # State
        self.placed_services = {} 
        self.queues: dict[str, deque[Task]] = {}
        self.backlogs = {}        
        self.service_profiles = {}
        self.slot_arrival_workload = {}
        self.last_cpu_allocations = {}
        
        self.used_ram = 0.0
        self.used_hdd = 0.0
        
        self.t_queue_max = {}
        self.frame_delay_history = defaultdict(list)
        
        # Solver
        self.solver = KKTSolver(self.cpu_capacity, learning_rate=0.05, max_iter=30)

    def reset(self):
        self.placed_services = {}
        self.queues = {}
        self.backlogs = {}
        self.slot_arrival_workload = {}
        self.last_cpu_allocations = {}
        self.used_ram = 0.0
        self.used_hdd = 0.0

    def update_placement(self, placement_vector, service_profiles):
        """
        Cập nhật Placement cho Frame mới.
        Logic: Nếu service bị gỡ, chỉ cần không đưa vào `placed_services`.
               Hàng đợi cũ (queues, backlogs) giữ nguyên, không xóa, không drop.
        """
        self.service_profiles = {p['id']: p for p in service_profiles}
        
        # 1. Reset trạng thái tài nguyên của Node cho Frame mới
        # Lưu ý: Ta KHÔNG reset self.queues hay self.backlogs 
        # để giữ lại task tồn đọng cho các frame sau.
        self.placed_services = {}
        self.used_ram = 0.0
        self.used_hdd = 0.0
        
        constraint_violations = 0
        
        # 2. Duyệt vector quyết định mới
        for svc_id, decision in enumerate(placement_vector):
            if decision == 1:
                # Nếu quyết định đặt service, thử deploy
                profile = service_profiles[svc_id]
                success = self._deploy_single_service(profile)
                if not success:
                    constraint_violations += 1
            else:
                # Nếu decision == 0 (Gỡ hoặc không đặt):
                # Ta đơn giản là không gọi _deploy_single_service.
                # Service ID này sẽ không có trong self.placed_services.
                # Hàng đợi self.queues[svc_id] (nếu có) sẽ bị "đóng băng" tại đây.
                pass
                
        return constraint_violations

    def _deploy_single_service(self, profile):
        omega = profile['omega']
        size_gb = profile['size'] / 1024.0
        can_deploy = False
        
        if omega == 1 and self.used_ram + size_gb <= self.ram_capacity:
            self.used_ram += size_gb
            can_deploy = True
        elif omega == 0 and self.used_hdd + size_gb <= self.hdd_capacity:
            self.used_hdd += size_gb
            can_deploy = True
        
        if can_deploy:
            svc_id = profile['id']
            self.placed_services[svc_id] = True
            
            if svc_id not in self.queues:
                self.queues[svc_id] = deque()
                self.backlogs[svc_id] = 0.0
                
            # --- Ước lượng T_queue_max động ---
            if svc_id not in self.t_queue_max:
                # 1. Lấy Deadline từ profile (ví dụ 0.1s)
                deadline = profile.get('deadline', 0.1) 
                
                # 2. Ước lượng thời gian truyền (T_trans)
                # Giả sử ta có hằng số băng thông trung bình AVG_BW (bits/s)
                # input_size đơn vị bits
                avg_bandwidth = 50 * 1024 * 1024 * 8 # Ví dụ 50 MB/s ~ 400 Mbps
                input_size = profile.get('data_size', 1 * 1024 * 1024 * 8) # Ví dụ 1MB
                est_t_trans = input_size / avg_bandwidth
                
                # 3. Ước lượng thời gian xử lý (T_proc)
                # Lấy workload của model trung bình (hoặc model mặc định đầu tiên)
                # Đơn vị workload: GFLOPS
                workload = sum(pr['workload'] for pr in profile['models'])*20/ len(profile['models']) if len(profile['models']) else 0.0
                
                # Năng lực tính toán của node: self.cpu_capacity (GFLOPS)
                # nên ta ước lượng nó chỉ được dùng khoảng 50-80% capacity
                est_cpu_share = self.cpu_capacity * 0.5
                est_t_proc = workload / max(est_cpu_share, 1e-9)
                
                # 4. Tính thời gian dư (Slack time)
                slack_time = deadline - est_t_trans - est_t_proc
                
                # 5. Gán giá trị khởi tạo an toàn
                if slack_time > 0:
                    # Lấy 50% thời gian dư làm ngân sách hàng đợi
                    self.t_queue_max[svc_id] = slack_time * 0.5 
                else:
                    # Trường hợp node quá yếu, thời gian xử lý đã > deadline
                    # Gán một giá trị rất nhỏ để ép f_min tăng tối đa
                    self.t_queue_max[svc_id] = 5e-3
            return True
        return False

    def admit_task(self, task: Task):
        sid = task.service_id
        if not self.placed_services.get(sid, False): return False
        if task.required_workload_gflops <= 0: return False
            
        self.queues[sid].append(task)
        
        # Cập nhật Backlog tổng
        self.backlogs[sid] = QueueDynamics.update_backlog(
            current_backlog=self.backlogs[sid],
            processed_workload=0,
            arrival_workload=task.required_workload_gflops
        )
        
        # Tracking arrival in slot to calculate Z 
        if sid not in self.slot_arrival_workload:
            self.slot_arrival_workload[sid] = 0.0
        self.slot_arrival_workload[sid] += task.required_workload_gflops
        
        return True

    def get_observation(self):
        return list(self.backlogs.values())

    def process_timeslot(self, current_time_elapsed, slot_duration, V_param=cfg.lypa_coef):
        """
        Hàm chính điều phối quy trình xử lý 1 Time Slot.
        """
        # 1. Lọc các service đang active
        active_svcs = [sid for sid, active in self.placed_services.items() if active]
        if not active_svcs:
            # print(f"  [Node {self.id}] No active services to process.")
            return [], 0.0
        
        print(f"\n--- [NODE {self.id}] TIMESLOT LOG (Duration: {slot_duration}s) ---")
        print(f"    Available CPU: {self.cpu_capacity} GFLOPS")

        # 2. [PLANNING] Tính toán phân bổ tài nguyên tối ưu (KKT Solver)
        f_alloc_vec, slot_cold_times = self._compute_optimal_resources(
            active_svcs, current_time_elapsed, slot_duration, V_param
        )

        # 3. [ACTION] Thực thi phân bổ, tính năng lượng và xử lý task
        completed_tasks, total_energy = self._execute_allocation(
            active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed
        )
        
        # 4. Reset các biến tracking tạm thời của slot
        self.slot_arrival_workload = {} 

        return completed_tasks, total_energy

    def _compute_optimal_resources(self, active_svcs, current_time_elapsed, slot_duration, V_param):
        """
        Bước 1: Tính toán các vector tham số và giải bài toán tối ưu KKT.
        Trả về: Vector phân bổ f và danh sách thời gian cold-start (nếu có).
        """
        num_active = len(active_svcs)
        G_vec = np.zeros(num_active)
        Z_vec = np.zeros(num_active)
        f_min_vec = np.zeros(num_active)
        f_max_vec = np.zeros(num_active)
        
        slot_cold_times = {}

        for i, sid in enumerate(active_svcs):
            profile = self.service_profiles[sid]
            omega = profile['omega']
            
            # G(tau) = Q(tau) (Backlog hiện tại)
            G_vec[i] = self.backlogs.get(sid, 0.0)
            
            # Z(tau) là trọng số phạt năng lượng (V * epsilon_c)
            Z_vec[i] = V_param * self.energy_coeff 
            
            # Đảm bảo Z không quá nhỏ để tránh lỗi số học trong Solver
            Z_vec[i] = max(Z_vec[i], 1e-12) 
            
            
            # f_max = Dung lượng node
            f_max_vec[i] = self.cpu_capacity
            
            # --- Tính f_min ---
            max_f_min_req = 0.0
            
            # Sinh thời gian cold-start ngẫu nhiên cho slot này (nếu cần)
            t_cold_val = 0.0
            if omega == 0:
                eps_cold= cfg.network["cold_start_time"]
                t_cold_val = random.uniform(eps_cold["min"], eps_cold["max"])
            slot_cold_times[sid] = t_cold_val

            # Duyệt hàng đợi để tìm yêu cầu khắt khe nhất
            current_t_q_max = self.t_queue_max.get(sid, 0.05)
            
            for task in list(self.queues[sid]):
                birth_time = task.created_at * slot_duration
                time_spent = current_time_elapsed - birth_time
                time_remaining = task.deadline - time_spent - t_cold_val - current_t_q_max
                
                req_f = 0.0
                if time_remaining <= 1e-8:
                    req_f = self.cpu_capacity  # Đã hết giờ -> Cần max tốc độ
                else:
                    # Lấy workload của task
                    if len(profile["models"]) > task.selected_model_idx:
                        task_workload = task.batch_size * profile["models"][task.selected_model_idx]["workload"]
                        req_f = task_workload / time_remaining # f = Workload / Time
                
                if req_f > max_f_min_req:
                    max_f_min_req = req_f
            
            # f_min không bao giờ được vượt quá năng lực thực tế của node
            f_min_vec[i] = min(max_f_min_req, self.cpu_capacity)
            
            if max_f_min_req > self.cpu_capacity:
                 # Ghi nhận trường hợp quá tải vật lý (để Debug nếu cần)
                 pass

        # Gọi Solver
        f_alloc_vec = self.solver.solve(G_vec, Z_vec, f_min_vec, f_max_vec)
        
        return f_alloc_vec, slot_cold_times

    def _execute_allocation(self, active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed):
        """
        Thực thi phân bổ, tính toán năng lượng, và kiểm tra QoS cho các task hoàn thành.
        Thêm tham số: current_time_elapsed (để tính thời điểm hoàn thành).
        """
        total_energy = 0.0
        completed_tasks = []
        
        # Lưu lại f để dùng cho observation state
        self.last_cpu_allocations = {}

        for i, sid in enumerate(active_svcs):
            f_val = f_alloc_vec[i]
            self.last_cpu_allocations[sid] = f_val
            
            profile = self.service_profiles[sid]
            omega = profile['omega']
            t_cold = slot_cold_times.get(sid, 0.0)
            
            # 1. Tính Năng Lượng (Giữ nguyên logic cũ)
            is_running = f_val > 1e-6
            e_cold_start = 0.0
            if omega == 0 and is_running:
                energy_param= cfg.network["energy"]
                epsilon_cold = energy_param.get('cold_start', 0.2)
                e_cold_start = epsilon_cold * t_cold
            
            e_comp = self.energy_coeff * (f_val**2) * slot_duration + e_cold_start
            total_energy += e_comp

            # 2. Xử lý Workload và Hàng đợi
            processed_workload = f_val * slot_duration
            q_before = self.backlogs.get(sid, 0.0)
            self.backlogs[sid] = max(0.0, q_before - processed_workload)
            q_after = self.backlogs[sid]
            
            # Log chi tiết cho từng service
            print(f"    [Svc {sid:2}] CPU: {f_val:8.2f} GFLOPS ({f_val/self.cpu_capacity*100:5.1f}%) | "
                  f"Energy: {e_comp:8.4f} J (Cold: {e_cold_start:5.2f}) | "
                  f"Queue: {q_before:8.2f} -> {q_after:8.2f} GFLOPS")
            
            remaining_cap = processed_workload
            
            while self.queues[sid] and remaining_cap > 0:
                task = self.queues[sid][0]
                
                # Đảm bảo task có biến theo dõi workload gốc và workload còn lại
                if not hasattr(task, 'remaining_workload'):
                     task.remaining_workload = task.required_workload_gflops
                if not hasattr(task, 'initial_workload'):
                     task.initial_workload = task.required_workload_gflops

                if task.remaining_workload <= remaining_cap:
                    # --- TASK HOÀN THÀNH ---
                    remaining_cap -= task.remaining_workload
                    task.remaining_workload = 0
                    done_task = self.queues[sid].popleft()
                    
                    # --- A. XÁC ĐỊNH TASK ĐÚNG HẠN KHÔNG? ---
                    # Thời điểm task xong = Thời điểm bắt đầu slot + thời lượng slot
                    finish_time = current_time_elapsed + slot_duration
                    
                    # Tổng thời gian task tồn tại từ lúc sinh ra đến lúc xong
                    total_stay_time = finish_time - done_task.created_at
                    
                    # Kiểm tra QoS (Deadline là khoảng thời gian cho phép, ví dụ 0.1s)
                    if total_stay_time <= done_task.deadline:
                        done_task.qos_status = True  # ĐẠT (Success)
                    else:
                        done_task.qos_status = False # TRƯỢT (Fail - QoS Violation)
                        
                    # --- B. GHI NHẬN DELAY ĐỂ UPDATE T_QUEUE_MAX ---
                    # Mục tiêu: Tính ra thời gian thực tế task phải "XẾP HÀNG"
                    # T_queue = T_total - T_processing - T_cold
                    
                    # Ước lượng thời gian xử lý lý thuyết (Workload / f_allocated)
                    # (Lưu ý: Nếu f biến thiên qua các slot, đây chỉ là ước lượng gần đúng tại slot cuối
                    # nhưng đủ tốt cho Gradient Descent)
                    est_proc_time = 0.0
                    if f_val > 1e-6:
                        est_proc_time = done_task.initial_workload / f_val
                    
                    observed_queue_delay = total_stay_time - est_proc_time - t_cold
                    
                    # Đảm bảo không âm (do sai số ước lượng)
                    observed_queue_delay = max(0.0, observed_queue_delay)
                    
                    # Lưu vào lịch sử để cuối Frame tính trung bình
                    self.frame_delay_history[sid].append(observed_queue_delay)
                    
                    completed_tasks.append(done_task)
                else:
                    # Task chưa xong, trừ bớt workload rồi dừng (đợi slot sau)
                    task.remaining_workload -= remaining_cap
                    remaining_cap = 0
                    
        print(f"    => Node Total Energy: {total_energy:.4f} J | Tasks Finished: {len(completed_tasks)}")
        return completed_tasks, total_energy
    
    def get_observation_state(self, service_id):
        """
        Trả về trạng thái cụ thể cho 1 service (Eq. 51):
        1. Queue Backlog (Q)
        2. Last CPU Allocation (f)
        """
        q = self.backlogs.get(service_id, 0.0)
        f = self.last_cpu_allocations.get(service_id, 0.0)
        return [q, f]