import gymnasium as gym
import numpy as np
import yaml
import os
from typing import Dict, List, Any

from src.network import TopologyManager
from src.entities import ComputingNode
from src.core.workload_generator import WorkloadGenerator
from src.core.time_manager import TimeManager
from src.utils.monitor import SimulationMonitor
from src.utils.config_loader import cfg
from src.entities.task import Task
from src import hp

class HMFD3QNEnv(gym.Env):
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_configs()
        
        # 1. Quản lý Thời gian
        self.time_manager = TimeManager(self.sim_config['time'])
        
        # 2. Mạng & Nodes
        self.topo = TopologyManager(cfg.TOPOLOGY_JSON, "configs/network_params.yaml")
        self.nodes: Dict[int, ComputingNode] = {}
        self._init_nodes()
        computing_node_ids = list(self.nodes.keys())
        
        # get TopologyManager
        self.logical_neighbors = self.topo.get_logical_neighbors(
            computing_node_ids, 
            max_hops=3
        )
        # 3. Workload
        terminals = self.topo.get_nodes_by_type('edge')
        self.workload_gen = WorkloadGenerator(
            self.sim_config['workload'],
            self.service_config['services'],
            terminals
        )
        
        # 4. Task Buffer (QUAN TRỌNG: Lưu task chờ xử lý trong slot này)
        self.task_buffer: List[Task] = []
        
        # 5. Monitor & Metrics
        self.monitor = SimulationMonitor(log_dir=cfg.LOGS_DIR)
        self.total_energy = 0.0
        self.qos_violations = 0
        self.completed_tasks = 0

    def _load_configs(self):
        with open("configs/simulation.yaml") as f: self.sim_config = yaml.safe_load(f)['simulation']
        with open("configs/services.yaml") as f: self.service_config = yaml.safe_load(f)

    def _init_nodes(self):
        node_ids = self.topo.get_all_node_ids()
        with open("configs/network_params.yaml") as f:
            net_params = yaml.safe_load(f)
            
        for nid in node_ids:
            topo_node = self.topo.graph.nodes[nid]
            ntype = topo_node.get('type', 'relay')
            if ntype == 'relay':
                continue

            specs = net_params['nodes'].get(ntype, net_params['nodes']['edge']).copy()
            specs['type'] = ntype
            specs['id'] = nid
            self.nodes[nid] = ComputingNode(nid, specs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_manager.reset()
        
        # Reset Metrics & Nodes
        self.total_energy = 0.0
        self.qos_violations = 0
        self.completed_tasks = 0
        for node in self.nodes.values():
            node.reset()
            
        # init task for first Slot 0
        self.task_buffer = self.workload_gen.generate(0)
        
        return self._get_observation(), {"new_tasks": self.task_buffer}

    def step(self, lower_actions: Dict, upper_actions: Dict = None):
        """
        lower_actions: Dict {task_id: {'node': int, 'model': int}}
                       Hành động cho các task nằm trong self.task_buffer
        """
        step_energy = 0.0
        step_qos_violation = 0
        
        # =================================================================
        # 1. UPPER LAYER (Service Placement)
        # =================================================================
        if self.time_manager.is_new_frame() and upper_actions:
            # print(f"[Sim] New Frame {self.time_manager.current_frame}. Updating Placement.")
            for nid, placement_vec in upper_actions.items():
                if nid in self.nodes:
                    violations = self.nodes[nid].update_placement(
                        placement_vec, 
                        self.service_config['services']
                    )
                    # Có thể phạt thêm nếu placement vi phạm tài nguyên (optional)

        # =================================================================
        # 2. LOWER LAYER (Task Offloading cho Tasks trong Buffer)
        # =================================================================
        # Xử lý các task đã được sinh ra từ step trước (hoặc reset)
        current_tasks = self.task_buffer
        
        for task in current_tasks:
            action = lower_actions.get(task.id)
            
            # Nếu Agent không đưa ra hành động cho task này -> Drop & Vi phạm
            if not action:
                step_qos_violation += 1
                continue
                
            target_node = action['node']
            model_idx = action['model']
            
            # A. Kiểm tra Model Selection
            svc_profile = self.service_config['services'][task.service_id]
            selected_model = next((m for m in svc_profile['models'] if m['id'] == model_idx), None)
            
            if not selected_model or selected_model['accuracy'] < task.min_accuracy:
                step_qos_violation += 1
                continue # Acc không đạt -> Drop

            # Cập nhật Workload (Eq. 12)
            # task.batch_size tương ứng với tỉ lệ d/D
            task.required_workload = selected_model['workload'] * task.batch_size
            task.selected_model_idx = model_idx

            # B. Tính toán Mạng (Truyền dẫn)
            trans_delay, hops, _ = self.topo.get_path_metrics(
                task.terminal_id, target_node, task.size
            )
            
            if trans_delay == float('inf'):
                step_qos_violation += 1
                continue

            # C. Năng lượng truyền dẫn (Eq. 11)
            # Giả sử cfg.TRANSMISSION_POWER (Watt)
            e_trans = cfg.TRANSMISSION_POWER * trans_delay 
            step_energy += e_trans
            
            # D. Đẩy vào Node (Admission)
            # Thời điểm đến = Hiện tại + Trễ truyền
            task.arrival_time_at_node = self.time_manager.time_elapsed + trans_delay
            
            if target_node in self.nodes:
                admitted = self.nodes[target_node].admit_task(task)
                if not admitted:
                    step_qos_violation += 1 # Node reject (Full queue hoặc chưa place service)
            else:
                step_qos_violation += 1

        # =================================================================
        # 3. PHYSICAL PROCESSING (Nodes chạy KKT)
        # =================================================================
        for node in self.nodes.values():
            done_tasks, node_energy = node.process_timeslot(
                current_time_elapsed=self.time_manager.time_elapsed,
                slot_duration=self.time_manager.slot_duration,
                V_param=hp.V
            )
            
            step_energy += node_energy
            self.completed_tasks += len(done_tasks)
            
            # Kiểm tra Deadline cho các task đã xong
            # Lưu ý: total_delay bao gồm cả thời gian chờ trong queue (đã được process_timeslot tính toán ngầm qua thời gian thực)
            for t in done_tasks:
                # Thời điểm sinh ra thực tế
                birth_time = t.created_at * self.time_manager.slot_duration
                # Thời điểm hoàn thành (xấp xỉ cuối slot hiện tại)
                # Để chính xác hơn, có thể lưu finish_time trong Task object tại ComputingNode
                finish_time = getattr(t, 'finish_time', self.time_manager.time_elapsed + self.time_manager.slot_duration)
                
                total_delay = finish_time - birth_time
                if total_delay > t.deadline:
                    step_qos_violation += 1

        # =================================================================
        # 4. UPDATE STATE & GENERATE NEXT TASKS
        # =================================================================
        self.total_energy += step_energy
        self.qos_violations += step_qos_violation
        
        # Advance Time
        self.time_manager.tick()
        
        # Sinh task cho slot TIẾP THEO (t+1)
        # Agent sẽ nhận task này trong 'info' hoặc 'obs' để quyết định cho bước sau
        next_tasks = self.workload_gen.generate(self.time_manager.current_slot)
        self.task_buffer = next_tasks # Cập nhật buffer
        
        # Reward Function (Eq. 53)
        # Alpha/Beta weights nên load từ config
        w_q1 = 5.0 
        reward = -(step_energy + w_q1 * step_qos_violation)
        
        done = self.time_manager.is_done()
        
        info = {
            "slot": self.time_manager.current_slot,
            "energy": step_energy,
            "qos_violations": step_qos_violation,
            "completed": self.completed_tasks,
            "new_tasks": next_tasks, # Quan trọng: Input cho Agent ở step sau
            "is_new_frame": self.time_manager.is_new_frame()
        }
        
        self.monitor.log_step(self, self.time_manager.current_slot, info)
        
        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        """
        State Space:
        1. Backlogs của các Node (System State).
        2. Thông tin Task trong Buffer (Request State) - Tùy thuộc thiết kế Agent.
           Nếu dùng MARL, mỗi Terminal Agent sẽ chỉ nhìn thấy task của nó.
        """
        # Node states
        node_states = {}
        for nid, node in self.nodes.items():
            node_states[nid] = node.get_observation()
            
        # Có thể trả về thêm task_buffer nếu Agent là Centralized
        # Hoặc Agent sẽ tự lấy 'new_tasks' từ info để xử lý.
        return node_states