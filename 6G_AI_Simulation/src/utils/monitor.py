import json
import os
import csv
import numpy as np

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

    def log_step(self, env, step: int, info: dict):
        """
        Ghi log trạng thái hiện tại của Environment.
        """
        # 1. Trích xuất dữ liệu Nodes (Tất cả nodes trong topology)
        nodes_data = []
        total_backlog = 0
        total_capacity = 0
        
        # Lấy tất cả node từ topology graph để hiển thị toàn bộ
        for nid, data in env.topo_manager.graph.nodes(data=True):
            node = env.nodes.get(nid)
            
            if node:
                # Node có khả năng tính toán (Cloud, Edge, Network)
                current_load = sum(node.backlogs.values())
                capacity = node.cpu_capacity
                utilization = min(current_load / capacity, 1.0) if capacity > 0 else 0
                active_services = [int(sid) for sid, active in node.placed_services.items() if active]
                energy = getattr(node, 'last_energy', 0.0)
                
                total_backlog += current_load
                total_capacity += capacity
            else:
                # Node trung chuyển hoặc Terminal (nếu có trong graph)
                current_load = 0
                capacity = 0
                utilization = 0
                active_services = []
                energy = 0
            
            nodes_data.append({
                "id": nid,
                "type": data.get('type', 'relay'),
                "cpu_util": utilization,
                "backlog": current_load,
                "capacity": capacity,
                "energy": energy,
                "active_services": active_services
            })

        # 2. Trích xuất Topology (Links)
        links_data = []
        for u, v in env.topo_manager.graph.edges():
            links_data.append({"source": u, "target": v})

        # 3. Tạo Snapshot JSON
        snapshot = {
            "step": step,
            "global_energy": env.total_energy,
            "qos_violations": env.total_violations,
            "completed_tasks": env.total_completed_tasks,
            "nodes": nodes_data,
            "links": links_data,
            "last_reward": info.get('reward', 0)
        }
        
        # Ghi đè file JSON (Atomic write để tránh lỗi khi Dashboard đang đọc)
        temp_file = self.state_file + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(snapshot, f)
            # Trên Windows, os.replace có thể lỗi nếu dashboard đang đọc file
            if os.path.exists(self.state_file):
                try:
                    os.replace(temp_file, self.state_file)
                except PermissionError:
                    pass # Bỏ qua bước này nếu file đang bị lock, bước sau sẽ cập nhật tiếp
            else:
                os.rename(temp_file, self.state_file)
        except Exception:
            pass

        # 4. Ghi nối file CSV
        avg_util = total_backlog / total_capacity if total_capacity > 0 else 0
        with open(self.history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, 
                env.total_energy, 
                env.total_violations, 
                env.total_completed_tasks,
                avg_util
            ])