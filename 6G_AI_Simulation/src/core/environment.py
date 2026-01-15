import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque

from src.utils import cfg
from src.network.topology_manager import TopologyManager
from src.network.channel_model import ChannelModel
from src.entities.node import ComputingNode
from src.entities.terminal import Terminal
from src.core.workload_generator import WorkloadGenerator
from src.core.time_manager import TimeManager

class SixGEnvironment:
    def __init__(self, service_config: List[Dict]):
        """
        Môi trường mô phỏng mạng 6G tối ưu hóa AI Service Placement & Task Scheduling.
        Dựa trên bài báo: Joint AI Service Placement, Task Scheduling, and Resource Allocation for IoT in 6G.
        """
        self.service_config = service_config
        self.num_services = len(service_config)
        self.device = cfg.device

        # 1. Khởi tạo thành phần mạng
        self.topo_manager = TopologyManager()
        self.topo_manager.load_topology_from_data()
        
        self.channel_model = ChannelModel(config=cfg.network)
        self.channel_model.topo = self.topo_manager 

        self.nodes: Dict[str, ComputingNode] = {}
        self.agent_node_ids = []  # Các node tham gia học (Edge/Network)
        self.cloud_node_ids = []  # Các node Cloud (Full services, không học)
        self._init_nodes()

        self.terminals: Dict[str, Terminal] = {}
        self._init_terminals()
        
        self.workload_gen = WorkloadGenerator(
            workload_config=cfg.task_param,
            service_config=service_config,
            terminals=list(self.terminals.values())
        )

        self.time_manager = TimeManager()

        # 2. Thiết lập Kích thước (Dimensions) cho Agent
        self.node_id_to_idx = {nid: i for i, nid in enumerate(sorted(self.nodes.keys()))}
        self.num_nodes_total = len(self.nodes)
        self.max_models_total = max([len(svc['models']) for svc in service_config])
        
        # Upper Agent State/Action Dims
        self.upper_state_dim = 2 * self.num_services # [Placement_prev, Popularity]
        self.upper_action_dim = self.num_services    # Multi-binary hoặc Discrete map
        
        # Lower Agent State/Action Dims
        self.lower_state_dim = 4 + (2 * self.num_nodes_total) # [Task_info(4), Node_states(2*N)]
        self.lower_action_dim = self.num_nodes_total * self.max_models_total
        self.mf_lower_dim = self.num_nodes_total + self.max_models_total

        # 3. Tracking & Metrics
        self.frame_F1_accumulation = 0.0
        self.total_completed_tasks = 0
        self.node_request_history = defaultdict(lambda: np.zeros(self.num_services))
        self.last_terminal_actions = defaultdict(lambda: np.zeros(self.mf_lower_dim, dtype=np.float32))

    def _init_nodes(self):
        """Khởi tạo node và áp dụng logic Cloud Node."""
        for node_id, data in self.topo_manager.graph.nodes(data=True):
            if data.get('type') in ['edge', 'cloud', 'network']:
                specs = {
                    'cpu': data['cpu_available'],
                    'ram': data['ram_capacity'],
                    'hdd': data['hdd_capacity'],
                    'energy_coeff': data['energy_coef'],
                    'type': data['type']
                }
                node = ComputingNode(node_id, specs)
                
                if data.get('type') == 'cloud':
                    # Cloud nodes mặc định bật toàn bộ dịch vụ (Không cần agent học)
                    full_placement = [1] * self.num_services
                    node.update_placement(full_placement, self.service_config)
                    self.cloud_node_ids.append(node_id)
                else:
                    self.agent_node_ids.append(node_id)
                
                self.nodes[node_id] = node

    def _init_terminals(self):
        edge_nodes = self.topo_manager.get_nodes_by_type('edge')
        num_terminals = cfg.neuron_net['NUM_LOWER_AGENTS']
        for i in range(num_terminals):
            t_id = f"UE_{i}"
            assigned_edge = edge_nodes[i % len(edge_nodes)]
            self.terminals[t_id] = Terminal(t_id, assigned_edge, cfg.task_param['arrival_rate'])

    def reset(self):
        """Reset môi trường cho Episode mới."""
        self.time_manager.reset()
        for node in self.nodes.values():
            node.reset()
        
        # Re-enforce Cloud nodes placement
        full_vec = [1] * self.num_services
        for cid in self.cloud_node_ids:
            self.nodes[cid].update_placement(full_vec, self.service_config)

        self.frame_F1_accumulation = 0.0
        self.total_completed_tasks = 0
        self.node_request_history.clear()
        self.last_terminal_actions.clear()

        self.workload_gen.step(current_time_slot=0)
        return self._get_upper_obs()

    # =========================================================================
    #  UPPER LEVEL: AI Service Placement (Frame t)
    # =========================================================================

    def step_upper(self, agent_placement_actions: Dict[str, List[int]]):
        """
        Nhận hành động Placement từ các Upper Agents.
        """
        self.frame_F1_accumulation = 0.0
        self.node_request_history.clear() # Reset popularity cho frame mới

        # Cập nhật cho Agent Nodes
        for node_id in self.agent_node_ids:
            if node_id in agent_placement_actions:
                self.nodes[node_id].update_placement(agent_placement_actions[node_id], self.service_config)

    def get_upper_feedback(self):
        """Trả về tuple (Next_Obs, Next_MF, Reward) cho Upper Agents sau khi hết Frame."""
        # Eq. 40: Reward_upper = -Sum(F1). Scale 1e-8 để ổn định training.
        reward = -self.frame_F1_accumulation * 1e-8
        
        all_obs, all_mf = self._get_upper_obs()
        
        # Chỉ trả về cho các node có Agent học
        agent_obs = {nid: all_obs[nid] for nid in self.agent_node_ids}
        agent_mf = {nid: all_mf[nid] for nid in self.agent_node_ids}
        
        return agent_obs, agent_mf, reward

    def _get_upper_obs(self):
        """Eq. 34-37: State của Upper Agent."""
        observations, mean_fields = {}, {}
        
        # Lấy trạng thái placement hiện tại của toàn mạng
        all_placements = {nid: np.array([1.0 if n.placed_services.get(i, False) else 0.0 
                          for i in range(self.num_services)]) for nid, n in self.nodes.items()}

        for node_id, node in self.nodes.items():
            # Local State: [x_v, phi_v]
            total_reqs = max(1.0, np.sum(self.node_request_history[node_id]))
            phi = self.node_request_history[node_id] / total_reqs
            observations[node_id] = np.concatenate([all_placements[node_id], phi]).astype(np.float32)
            
            # Mean Field: Trung bình Placement của các node láng giềng vật lý
            neighbors = [n for n in self.topo_manager.graph.neighbors(node_id) if n in self.nodes]
            if neighbors:
                mean_fields[node_id] = np.mean([all_placements[n] for n in neighbors], axis=0)
            else:
                mean_fields[node_id] = np.zeros(self.num_services, dtype=np.float32)
                
        return observations, mean_fields

    # =========================================================================
    #  LOWER LEVEL: Task Scheduling (Slot tau)
    # =========================================================================

    def step_lower(self, actions_map: Dict[str, int]):
        """
        Xử lý Scheduling trong 1 Time Slot.
        actions_map: {terminal_id: discrete_action_id}
        """
        slot_energy = 0.0
        slot_qos_violations = 0
        slot_arrival_tasks = 0
        arrivals_A = defaultdict(float)
        
        # 1. Decode Actions & Transmission
        self.last_terminal_actions.clear()
        node_list = sorted(self.nodes.keys())

        for tid, action_id in actions_map.items():
            # Decode: Action_ID -> (Node_Index, Model_Index)
            node_idx = action_id // self.max_models_total
            model_idx = action_id % self.max_models_total
            target_node_id = node_list[node_idx]
            
            # Tracking Action cho Mean Field
            action_vec = np.zeros(self.mf_lower_dim, dtype=np.float32)
            action_vec[node_idx] = 1.0
            action_vec[self.num_nodes_total + model_idx] = 1.0
            self.last_terminal_actions[tid] = action_vec
            
            # Logic xử lý task
            term = self.terminals[tid]
            task = term.current_task
            if task:
                slot_arrival_tasks += 1
                self.node_request_history[target_node_id][task.service_id] += 1
                
                if target_node_id in self.nodes:
                    node = self.nodes[target_node_id]
                    svc_profile = self.service_config[task.service_id]
                    
                    # Cấp phát Model & Workload (Đảm bảo model_idx hợp lệ cho service này)
                    actual_model_idx = model_idx % len(svc_profile['models'])
                    model_info = svc_profile['models'][actual_model_idx]
                    model_workload = model_info['workload'] * task.batch_size
                    task.assign_schedule(target_node_id, actual_model_idx, model_workload)
                    arrivals_A[(target_node_id, task.service_id)] += model_workload

                    # Transmission
                    meta = self.channel_model.get_metadata(term.edge_id, target_node_id, task.total_data_size_mb)
                    task.transmission_delay = meta['tranmission_delay']
                    slot_energy += meta['transmission_energy'] 
                    
                    if not node.admit_task(task):
                        slot_qos_violations += 1

        # 2. Node Processing (KKT Resource Allocation)
        current_abs_time = self.time_manager.to_abs_time(self.time_manager.current_slot)
        queues_before = {nid: node.backlogs.copy() for nid, node in self.nodes.items()}

        for node in self.nodes.values():
            done_tasks, n_energy = node.process_timeslot(current_abs_time, self.time_manager.slot_duration)
            slot_energy += n_energy
            self.total_completed_tasks += len(done_tasks)
            for t in done_tasks:
                if not t.qos_status: slot_qos_violations += 1

        # 3. Lyapunov F1 Calculation (Eq. 23)
        drift_term = 0.0
        for nid, node in self.nodes.items():
            for sid in range(self.num_services):
                Q = queues_before[nid].get(sid, 0.0)
                A = arrivals_A.get((nid, sid), 0.0)
                W = node.last_cpu_allocations.get(sid, 0.0) * self.time_manager.slot_duration
                drift_term += Q * (A - W)

        F1_tau = drift_term + (cfg.lypa_coef * slot_energy)
        self.frame_F1_accumulation += F1_tau

        # 4. Update & Feedback
        self.time_manager.tick()
        if not self.time_manager.is_done():
            self.workload_gen.step(self.time_manager.current_slot)

        next_obs, next_mf, _ = self._get_lower_obs()
        
        # Lower Reward (Eq. 53). Scale 1e-8.
        raw_rewards = self._calculate_lower_reward(F1_tau, slot_qos_violations)
        scaled_rewards = {tid: r * 1e-8 for tid, r in raw_rewards.items()}

        info = {
            "energy": slot_energy, "violations": slot_qos_violations,
            "F1_tau": F1_tau, "drift_term": drift_term,
            "arrival_tasks": slot_arrival_tasks,
            "is_new_frame": self.time_manager.is_new_frame()
        }
        
        # --- EXPORT DATA CHO DASHBOARD ---
        self.export_live_state(info)
        
        return next_obs, scaled_rewards, self.time_manager.is_done(), info

    def export_live_state(self, info):
        """Xuất trạng thái hệ thống ra JSON cho Streamlit Dashboard"""
        import json
        import os
        
        state = {
            "step": self.time_manager.current_slot,
            "global_energy": info['energy'],
            "qos_violations": info['violations'],
            "completed_tasks": self.total_completed_tasks,
            "incoming_tasks": info.get('arrival_tasks', 0),
            "nodes": [],
            "links": []
        }
        
        # 1. Trạng thái Node
        for nid, node in self.nodes.items():
            total_backlog = sum(node.backlogs.values())
            # Tính độ tải dựa trên backlog so với 1 ngưỡng (ví dụ 5000 GFLOPS)
            load_factor = min(1.0, total_backlog / 5000.0) 
            
            state["nodes"].append({
                "id": nid,
                "type": node.type,
                "cpu_util": load_factor,
                "backlog": total_backlog,
                "active_services": [sid for sid, active in node.placed_services.items() if active]
            })
            
        # 2. Topology Links
        for u, v in self.topo_manager.graph.edges():
            if u in self.nodes and v in self.nodes:
                state["links"].append({"source": u, "target": v})
                
        # Lưu file
        os.makedirs("data", exist_ok=True)
        with open("data/live_state.json", "w") as f:
            json.dump(state, f)
            
        # Lưu history để vẽ đồ thị
        history_file = "data/history.csv"
        import pandas as pd
        new_row = pd.DataFrame([{
            "step": state["step"],
            "total_energy": state["global_energy"],
            "qos_violations": state["qos_violations"]
        }])
        if not os.path.exists(history_file):
            new_row.to_csv(history_file, index=False)
        else:
            new_row.to_csv(history_file, mode='a', header=False, index=False)

    def _get_lower_obs(self):
        """Eq. 51: State của Lower Agent (Terminal) kèm Action Mask."""
        observations, mean_fields, masks = {}, {}, {}
        node_ids = sorted(self.nodes.keys())
        
        # Network State Snapshot
        net_snapshot = {}
        for sid in range(self.num_services):
            states = []
            for nid in node_ids:
                states.extend(self.nodes[nid].get_observation_state(sid))
            net_snapshot[sid] = np.array(states, dtype=np.float32)

        # Mean Field (Edge-based grouping)
        edge_groups = defaultdict(list)
        for tid, term in self.terminals.items(): edge_groups[term.edge_id].append(tid)
        
        for tid, term in self.terminals.items():
            task = term.current_task
            mask = np.ones(self.lower_action_dim, dtype=np.float32) # Mặc định cho phép (nếu không có task)
            
            if task is None:
                observations[tid] = np.zeros(self.lower_state_dim, dtype=np.float32)
            else:
                local = np.array([task.total_data_size_mb, task.deadline, float(task.omega), task.min_accuracy])
                observations[tid] = np.concatenate([local, net_snapshot[task.service_id]])
                
                # Tạo Mask: Chỉ cho phép các Node đã đặt Service này
                mask = np.zeros(self.lower_action_dim, dtype=np.float32)
                for n_idx, nid in enumerate(node_ids):
                    if self.nodes[nid].placed_services.get(task.service_id, False):
                        # Bật mask cho tất cả các model của node hợp lệ này
                        start_idx = n_idx * self.max_models_total
                        end_idx = start_idx + self.max_models_total
                        mask[start_idx:end_idx] = 1.0
                
                # Trường hợp xấu nhất (không node nào cài service): cho phép Cloud Nodes 
                # để tránh cộng tổng bằng 0 khi Softmax
                if np.sum(mask) == 0:
                    for n_idx, nid in enumerate(node_ids):
                        if nid in self.cloud_node_ids:
                            mask[n_idx * self.max_models_total : (n_idx+1) * self.max_models_total] = 1.0

            masks[tid] = mask
            
            # MF là trung bình hành động của các terminal trong cùng vùng Edge
            group = edge_groups[term.edge_id]
            mean_fields[tid] = np.mean([self.last_terminal_actions[gtid] for gtid in group], axis=0)

        return observations, mean_fields, masks

    def _calculate_lower_reward(self, F1_tau, violations):
        """Eq. 53: Reward = -[F1 + QoS_Penalty]"""
        w1, w2 = cfg.neuron_net.get("OMEGA_Q1", 1000), cfg.neuron_net.get("OMEGA_Q2", 0.12)
        penalty = w1 * np.exp(w2 * violations)
        reward = -(F1_tau + penalty)
        return {tid: reward for tid in self.terminals}