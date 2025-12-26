import json
import os
import csv
import numpy as np
from src.core import HMFD3QNEnv

class SimulationMonitor:
    def __init__(self, log_dir="data/"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.state_file = os.path.join(self.log_dir, "live_state.json")
        self.history_file = os.path.join(self.log_dir, "history.csv")
        
        # Khởi tạo file CSV với Header
        with open(self.history_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "total_energy", "qos_violations", "completed_tasks", "avg_cpu_util"])

    def log_step(self, env: HMFD3QNEnv, step: int, info: dict):
        """
        Ghi log trạng thái hiện tại của Environment.
        """
        # 1. Trích xuất dữ liệu Nodes
        nodes_data = []
        total_backlog = 0
        total_capacity = 0
        
        for nid, node in env.nodes.items():
            # Tính CPU Utilization %
            current_load = sum(node.backlogs.values())
            capacity = node.cpu_capacity
            utilization = min(current_load / capacity, 1.0) if capacity > 0 else 0
            
            total_backlog += current_load
            total_capacity += capacity
            
            # Danh sách dịch vụ đang active (Binary vector)
            # Giả sử có tối đa 10 service để hiển thị đẹp
            active_services = [int(sid) for sid, active in node.placed_services.items() if active]
            
            nodes_data.append({
                "id": nid,
                "type": node.type,
                "cpu_util": utilization,
                "backlog": current_load,
                "active_services": active_services
            })

        # 2. Trích xuất Topology (Links)
        links_data = []
        for u, v in env.topo.graph.edges():
            links_data.append({"source": u, "target": v})

        # 3. Tạo Snapshot JSON
        snapshot = {
            "step": step,
            "global_energy": env.total_energy,
            "qos_violations": env.qos_violations,
            "completed_tasks": env.completed_tasks,
            "nodes": nodes_data,
            "links": links_data,
            "last_reward": info.get('reward', 0)
        }
        
        # Ghi đè file JSON (Atomic write để tránh lỗi khi Dashboard đang đọc)
        temp_file = self.state_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(snapshot, f)
        os.replace(temp_file, self.state_file)

        # 4. Ghi nối file CSV
        avg_util = total_backlog / total_capacity if total_capacity > 0 else 0
        with open(self.history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, 
                env.total_energy, 
                env.qos_violations, 
                env.completed_tasks,
                avg_util
            ])