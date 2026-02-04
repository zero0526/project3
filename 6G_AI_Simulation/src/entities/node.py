from collections import deque, defaultdict
import numpy as np
import random
from typing import List, Dict
from src.mechanisms.kkt_solver import KKTSolver
from src.mechanisms.queue_dynamics import QueueDynamics
from src.mechanisms.energy_model import EnergyModel
from src.entities.task import Task
from src.utils import cfg

class ComputingNode:
    def __init__(self, node_id, specs, service_config: List[Dict], avg_hops: float = 1.0):
        self.id = node_id
        self.specs = specs
        self.cpu_capacity = specs['cpu']      # f_v(tau)
        self.ram_capacity = specs['ram']
        self.hdd_capacity = specs['hdd']
        self.energy_coeff = specs['energy_coeff'] # epsilon_c
        self.type= specs['type']
        
        self.avg_hops = avg_hops
        
        # State
        self.placed_services = {} 
        self.queues: dict[str, deque[Task]] = {}
        self.backlogs = {}        
        self.service_profiles = {}
        self.slot_arrival_workload = {}
        self.last_cpu_allocations = {}
        self.last_energy = 0.0        # <--- Added to track per-node consumption
        
        self.used_ram = 0.0
        self.used_hdd = 0.0
                
        # --- Khởi tạo t_queue_max cho toàn bộ các service ---
        self.t_queue_max = {}
        self.__init_deadline(service_config)

        # Solver - Tăng số vòng lặp để hội tụ tốt hơn
        self.solver = KKTSolver(self.cpu_capacity, learning_rate=0.05, max_iter=100)

    def __init_deadline(self, service_config):
        for svc in service_config:
            svc_id = svc['id']
            mean_deadline = svc.get('mean_deadline', 0.1)
            
            # Ước lượng thời gian truyền: 20 * data_bits * hops / bandwidth (50MB/s)
            data_bits = svc.get('input_data_size', 1.0) * 1024 * 1024 * 8
            bandwidth_bps = 50 * 1024 * 1024 * 8 
            est_t_trans = (10 * data_bits * self.avg_hops) / bandwidth_bps
            
            # Ước lượng thời gian xử lý: Model nhẹ nhất / 25% CPU
            if svc.get('models'):
                min_workload = min(m['workload'] for m in svc['models'])
            else:
                min_workload = 0.0
            
            est_cpu_share = self.cpu_capacity * 0.25
            est_t_proc = min_workload / max(est_cpu_share, 1e-9)
            
            # Tính Slack time và gán t_queue_max (kì vọng lan truyền/hàng đợi tối đa)
            slack_time = mean_deadline - est_t_trans - est_t_proc
            self.t_queue_max[svc_id] = 0.05

    def reset(self):
        self.placed_services = {}
        self.queues = {}
        self.backlogs = {}
        self.slot_arrival_workload = {}
        self.last_cpu_allocations = {}
        self.last_energy = 0.0
        self.used_ram = 0.0
        self.used_hdd = 0.0

    def update_placement(self, placement_vector, service_profiles):
        """
        Cập nhật Placement cho Frame mới.
        """
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
            else:
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
                
            return True
        return False

    def admit_task(self, task: Task):
        sid = task.service_id
        if not self.placed_services.get(sid, False): return False
        if task.required_workload_gflops <= 0: return False
            
        self.queues[sid].append(task)
        
        self.backlogs[sid] = QueueDynamics.update_backlog(
            current_backlog=self.backlogs[sid],
            processed_workload=0,
            arrival_workload=task.required_workload_gflops
        )
        
        if sid not in self.slot_arrival_workload:
            self.slot_arrival_workload[sid] = 0.0
        self.slot_arrival_workload[sid] += task.required_workload_gflops
        
        return True

    def process_timeslot(self, current_time_elapsed, slot_duration, V_param=cfg.lypa_coef):
        active_svcs = [sid for sid, active in self.placed_services.items() if active]
        if not active_svcs:
            return [], 0.0
        
        print(f"\n--- [NODE {self.id}] TIMESLOT LOG (Duration: {slot_duration}s) ---")
        print(f"    Available CPU: {self.cpu_capacity} GFLOPS")

        f_alloc_vec, slot_cold_times = self._compute_optimal_resources(
            active_svcs, current_time_elapsed, slot_duration, V_param
        )

        completed_tasks, total_energy = self._execute_allocation(
            active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed
        )
        
        self.slot_arrival_workload = {} 
        self.last_energy = total_energy

        return completed_tasks, total_energy

    def _compute_optimal_resources(self, active_svcs, current_time_elapsed, slot_duration, V_param):
        num_active = len(active_svcs)
        G_vec = np.zeros(num_active)
        Z_vec = np.zeros(num_active)
        f_min_vec = np.zeros(num_active)
        f_max_vec = np.zeros(num_active)
        
        slot_cold_times = {}

        for i, sid in enumerate(active_svcs):
            profile = self.service_profiles[sid]
            omega = profile['omega']
            
            G_vec[i] = self.backlogs.get(sid, 0.0)
            
            arrival_load = self.slot_arrival_workload.get(sid, 0.0)
            Z_vec[i] = V_param * self.energy_coeff * arrival_load
            Z_vec[i] = max(Z_vec[i], 1e-12) 
            
            f_max_vec[i] = self.cpu_capacity
            
            max_f_min_req = 0.0
            t_cold_val = 0.0
            if omega == 0:
                eps_cold= cfg.network["cold_start_time"]
                t_cold_val = random.uniform(eps_cold["min"], eps_cold["max"])
            slot_cold_times[sid] = t_cold_val

            current_t_q_max = self.t_queue_max.get(sid, 0.05)
            
            for task in list(self.queues[sid]):
                birth_time = task.created_at * slot_duration
                time_spent = current_time_elapsed - birth_time
                time_remaining = task.deadline - time_spent - t_cold_val - current_t_q_max
                
                req_f = 0.0
                if time_remaining <= 1e-8:
                    req_f = self.cpu_capacity
                else:
                    if len(profile["models"]) > task.selected_model_idx:
                        task_workload = task.batch_size * profile["models"][task.selected_model_idx]["workload"]
                        req_f = task_workload / time_remaining
                
                if req_f > max_f_min_req:
                    max_f_min_req = req_f
            
            f_min_vec[i] = min(max_f_min_req, self.cpu_capacity)

        f_alloc_vec = self.solver.solve(G_vec, Z_vec, f_min_vec, f_max_vec)
        
        return f_alloc_vec, slot_cold_times

    def _execute_allocation(self, active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed):
        total_energy = 0.0
        completed_tasks = []
        self.last_cpu_allocations = {}

        for i, sid in enumerate(active_svcs):
            f_val = f_alloc_vec[i]
            if not np.isfinite(f_val): f_val = 0.0
            
            self.last_cpu_allocations[sid] = f_val
            
            profile = self.service_profiles[sid]
            omega = profile['omega']
            t_cold = slot_cold_times.get(sid, 0.0)
            
            processed_workload = f_val * slot_duration
            q_before = self.backlogs.get(sid, 0.0)
            self.backlogs[sid] = max(0.0, q_before - processed_workload)
            q_after = self.backlogs[sid]
            
            actual_workload = q_before - q_after

            energy_param = cfg.network.get("energy", {})
            epsilon_cold = energy_param.get('cold_start', 0.2)
            
            effective_t_cold = t_cold if f_val > 1e-6 else 0.0
            
            e_comp = EnergyModel.calc_computation_energy(
                epsilon_c=self.energy_coeff,
                f_allocated=f_val,
                workload_total=actual_workload, 
                omega=omega,
                epsilon_cold=epsilon_cold,
                t_cold=effective_t_cold
            )

            total_energy += e_comp
            
            print(f"    [Svc {sid:2}] CPU: {f_val:8.2f} GFLOPS ({f_val/self.cpu_capacity*100:5.1f}%) | "
                  f"Energy: {e_comp:8.4f} J | "
                  f"Processed: {actual_workload:8.2f} GFLOPS | "
                  f"Queue: {q_before:8.2f} -> {q_after:8.2f} GFLOPS")
            
            remaining_cap = processed_workload
            
            while self.queues[sid] and remaining_cap > 0:
                task = self.queues[sid][0]
                
                if not hasattr(task, 'remaining_workload'):
                     task.remaining_workload = task.required_workload_gflops
                if not hasattr(task, 'initial_workload'):
                     task.initial_workload = task.required_workload_gflops

                if task.remaining_workload <= remaining_cap:
                    remaining_cap -= task.remaining_workload
                    task.remaining_workload = 0
                    done_task = self.queues[sid].popleft()
                    
                    time_in_slot = (processed_workload - remaining_cap) / f_val if f_val > 0 else 0
                    finish_time = current_time_elapsed + time_in_slot
                    
                    # Kiểm tra xem task đã bị quá hạn ngay từ lúc bắt đầu slot chưa để tối ưu tài nguyên
                    birth_time = done_task.created_at * slot_duration
                    if current_time_elapsed > birth_time + done_task.deadline:
                        # Task này đã chết rồii, không tính là thành công cho dù xử lý xong
                        done_task.qos_status = False
                    else:
                        done_task.qos_status = True
                    
                    done_task.finished_at = finish_time
                    completed_tasks.append(done_task)
                else:
                    task.remaining_workload -= remaining_cap
                    remaining_cap = 0
                    
        print(f"    => Node Total Energy: {total_energy:.4f} J | Tasks Finished: {len(completed_tasks)}")
        return completed_tasks, total_energy
    
    def get_observation_state(self, service_id):
        """
        Trả về trạng thái cụ thể cho 1 service (Eq. 51):
        1. Queue Backlog (Q) - Trả về raw để env tự normalize
        2. Last CPU Allocation (f) - Normalization 1/capacity
        """
        q = self.backlogs.get(service_id, 0.0)
        f = self.last_cpu_allocations.get(service_id, 0.0) / (self.cpu_capacity if self.cpu_capacity > 0 else 1.0)
        
        return [q, f]