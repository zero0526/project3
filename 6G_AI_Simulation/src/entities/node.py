from collections import deque
import numpy as np
import random
from src.mechanisms.kkt_solver import KKTSolver
from src.mechanisms.energy_model import EnergyModel
from src.mechanisms.queue_dynamics import QueueDynamics
from src.entities.task import Task
from src import hp

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
        self.service_profiles = {p['id']: p for p in service_profiles}
        self.placed_services = {}
        self.used_ram = 0.0
        self.used_hdd = 0.0
        
        constraint_violations = 0
        for svc_id, decision in enumerate(placement_vector):
            if decision == 1:
                profile = service_profiles[svc_id]
                success = self._deploy_single_service(profile)
                if not success:
                    constraint_violations += 1
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
            return True
        return False

    def admit_task(self, task: Task):
        sid = task.service_id
        if not self.placed_services.get(sid, False): return False
        if task.required_workload <= 0: return False
            
        self.queues[sid].append(task)
        
        # Cập nhật Backlog tổng
        self.backlogs[sid] = QueueDynamics.update_backlog(
            current_backlog=self.backlogs[sid],
            processed_workload=0,
            arrival_workload=task.required_workload
        )
        
        # Tracking arrival in slot to calculate Z 
        if sid not in self.slot_arrival_workload:
            self.slot_arrival_workload[sid] = 0.0
        self.slot_arrival_workload[sid] += task.required_workload
        
        return True

    def get_observation(self):
        return list(self.backlogs.values())

    def process_timeslot(self, current_time_elapsed, slot_duration, V_param=hp.V):
        """
        Xử lý tài nguyên tại Time Slot tau.
        """
        active_svcs = [sid for sid, active in self.placed_services.items() if active]
        if not active_svcs:
            return [], 0.0

        # --- 1. Chuẩn bị Vector G, Z, f_min, f_max ---
        num_active = len(active_svcs)
        G_vec = np.zeros(num_active)
        Z_vec = np.zeros(num_active)
        f_min_vec = np.zeros(num_active)
        f_max_vec = np.zeros(num_active)
        
        # Dict lưu t_cold sinh ra cho mỗi service trong slot này (để dùng đồng nhất)
        slot_cold_times = {} 

        for i, sid in enumerate(active_svcs):
            profile = self.service_profiles[sid]
            omega = profile['omega']
            
            # G(tau) = Q(tau)
            G_vec[i] = self.backlogs[sid]
            
            # Z(tau) = V * epsilon * arrival_Workload (Eq. 24)
            arrival_load = self.slot_arrival_workload.get(sid, 0.0) 
            Z_vec[i] = V_param * self.energy_coeff * max(arrival_load, 1e-6)
            
            # f_max
            f_max_vec[i] = self.cpu_capacity 
            
            # --- calc f_min (Eq. 25 & 18) ---
            # queue FIFO:
            # Task i completed when Task 1..i-1 completed.
            # Workload tích lũy = Sum(Workload 1..i)
            # Thời gian còn lại = Deadline_i - Trễ truyền - Trễ chờ - Cold start
            
            max_f_min_req = 0.0
            cumulative_workload = 0.0 # Tích lũy workload
            
            # Sinh t_cold cho service này nếu là Occasional
            t_cold_val = 0.0
            if omega == 0:
                t_cold_val = random.uniform(hp.EPS_COLD_T[0], hp.EPS_COLD_T[1])
            slot_cold_times[sid] = t_cold_val 

            # Duyệt task để tìm bottleneck
            # (Limit 20 task đầu để tối ưu hiệu năng)
            for task in list(self.queues[sid])[:20]: 
                # 1. Cộng dồn workload (Xử lý tuần tự)
                cumulative_workload += task.required_workload
                
                # 2. Tính thời gian đã trôi qua (Age of task)
                # created_at là số slot. current_time_elapsed là giây.
                birth_time = task.created_at * slot_duration
                time_spent = current_time_elapsed - birth_time
                
                # 3. Thời gian ngân sách còn lại (Remaining Budget)
                # Y(tau) bao gồm t_cold
                time_remaining = task.deadline - time_spent - t_cold_val
                
                if time_remaining <= 0.001: 
                    req_f = self.cpu_capacity # expired -> max
                else:
                    # Tốc độ để giải quyết (Workload Tích lũy) trong (Thời gian còn lại)
                    req_f = cumulative_workload / time_remaining
                
                # f_min 
                if req_f > max_f_min_req:
                    max_f_min_req = req_f
            
            f_min_vec[i] = max_f_min_req

        # --- 2. Gọi KKT Solver ---
        f_alloc_vec = self.solver.solve(G_vec, Z_vec, f_min_vec, f_max_vec)
        
        total_energy = 0.0
        completed_tasks = []

        # --- 3. Thực thi phân bổ & Tính Năng lượng ---
        for i, sid in enumerate(active_svcs):
            f_val = f_alloc_vec[i]
            
            profile = self.service_profiles[sid]
            omega = profile['omega']
            
            # [CORRECTED] Tính E_cold (Eq. 14)
            # E_cold = (1 - omega) * epsilon_cold * t_cold
            # Chỉ tốn năng lượng cold start nếu thực sự có chạy (f > 0)
            is_running = f_val > 1e-6
            
            e_cold_start = 0.0
            if omega == 0 and is_running:
                epsilon_cold = 0.2 # Watts (Table II)
                t_cold = slot_cold_times[sid]
                e_cold_start = epsilon_cold * t_cold
            
            # Tổng năng lượng = E_dynamic + E_cold
            e_comp = EnergyModel.calc_computation(
                coeff=self.energy_coeff,
                frequency=f_val,
                duration=slot_duration,
                omega=omega,
                cold_start_energy=e_cold_start # Truyền giá trị đã tính đúng
            )
            total_energy += e_comp
            
            # Cập nhật Backlog & Pop Queue
            processed_workload = f_val * slot_duration
            self.backlogs[sid] = QueueDynamics.update_backlog(self.backlogs[sid], processed_workload, 0)
            
            remaining_cap = processed_workload
            while self.queues[sid] and remaining_cap > 0:
                task = self.queues[sid][0]
                if not hasattr(task, 'remaining_workload'):
                    task.remaining_workload = task.required_workload
                
                if task.remaining_workload <= remaining_cap:
                    remaining_cap -= task.remaining_workload
                    done_task = self.queues[sid].popleft()
                    completed_tasks.append(done_task)
                else:
                    task.remaining_workload -= remaining_cap
                    remaining_cap = 0

        self.last_cpu_allocations = {} 
        for i, sid in enumerate(active_svcs):
            self.last_cpu_allocations[sid] = f_alloc_vec[i]
            
        self.slot_arrival_workload = {} 
        
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