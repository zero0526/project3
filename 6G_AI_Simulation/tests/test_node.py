import unittest
import numpy as np
import logging
import yaml
import sys
import os
import random
from collections import defaultdict

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.entities.node import ComputingNode
from src.network.topology_manager import TopologyManager
from src.entities.task import Task
from src.utils import cfg
from src.core.environment import SixGEnvironment

# Configure logging to output to console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class TestNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load settings
        cfg.topology_name = "atlanta"
        
        # Load Service Config
        service_path = os.path.join(project_root, "configs", "services.yaml")
        with open(service_path, 'r') as f:
            services_data = yaml.safe_load(f)
        cls.service_config = services_data['services']

        # Initialize Environment (This handles TopologyManager and Node initialization)
        cls.env = SixGEnvironment(cls.service_config)
        cls.topo_manager = cls.env.topo_manager

    def test_01_node_initialization_info(self):
        """
        Test 1: Khởi tạo mạng và log thông tin chi tiết từng node.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 1: NETWORK INITIALIZATION AND NODE LOGGING")
        logger.info("="*60)
        
        # Log thông tin các node đầu tiên từ môi trường
        node_ids = list(self.env.nodes.keys())[:5]
        for node_id in node_ids:
            node = self.env.nodes[node_id]
            node_data = self.topo_manager.get_node_resources(node_id)
            
            pos = node_data.get('pos', (0, 0))
            neighbors = list(self.topo_manager.graph.neighbors(node_id))
            
            logger.info(f"NODE ID: {node_id}")
            logger.info(f"  Type: {node.type}")
            logger.info(f"  Coordinates: {pos}")
            logger.info(f"  CPU Capacity: {node.cpu_capacity} GFLOPS")
            logger.info(f"  RAM Capacity: {node.ram_capacity} GB")
            logger.info(f"  Neighbors: {neighbors}")
            
            # Preview t_queue_max cho 3 services đầu tiên
            t_q_keys = list(node.t_queue_max.keys())[:3]
            t_q_preview = {k: f"{node.t_queue_max[k]:.4f}" for k in t_q_keys}
            logger.info(f"  t_queue_max (preview): {t_q_preview}")
            logger.info("-" * 40)
            
            self.assertIsNotNone(node.t_queue_max)
            self.assertIn(0, node.t_queue_max) # Check if service 0 is in deadline dict

    def test_02_kkt_resource_allocation_stability(self):
        """
        Test 2: Gửi task tăng dần đến 1 node từ các terminal ngẫu nhiên.
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 2: KKT RESOURCE ALLOCATION STABILITY UNDER INCREASING LOAD")
        logger.info("="*60)
        
        # 1. Chọn target node (loại edge) từ môi trường
        edge_nodes = [nid for nid, n in self.env.nodes.items() if n.type == 'edge']
        if not edge_nodes:
            edge_nodes = list(self.env.nodes.keys())
        target_nid = edge_nodes[0]
        node = self.env.nodes[target_nid]
        
        # 2. Đảm bảo có service được đặt (ví dụ service 0 và 1)
        placement_vector = [0] * len(self.service_config)
        placement_vector[0] = 1
        placement_vector[1] = 1
        node.update_placement(placement_vector, self.service_config)
        
        # 3. Lấy danh sách terminals từ môi trường
        terminals = list(self.env.terminals.values())
        
        # 4. Các mức độ load (Thấp: 10, Vừa: 50, Cao: 250 tasks)
        load_levels = {
            "LOW LOAD": 10,
            "MEDIUM LOAD": 50,
            "HIGH LOAD": 250
        }
        
        slot_duration = 0.1 # 100ms
        V_param = 1e5
        
        for level_name, task_count in load_levels.items():
            logger.info(f"\n>>> TESTING {level_name} ({task_count} tasks) <<<")
            
            # Reset trạng thái hàng đợi để test độc lập từng mức load
            for sid in node.queues:
                node.queues[sid].clear()
                node.backlogs[sid] = 0.0
            
            # Sinh task từ random terminals
            tasks_admitted = 0
            arrival_workload_gflops = defaultdict(float)
            for i in range(task_count):
                term = random.choice(terminals)
                task = term.step_generate_task(
                    current_time_slot=0,
                    arrival_rate=1.0, 
                    batch_size=20,
                    zipf_probs=self.env.workload_gen.zipf_probs,
                    service_config_list=self.service_config
                )
                
                if task:
                    if node.placed_services.get(task.service_id):
                        svc_info = self.service_config[task.service_id]
                        model_workload = svc_info['models'][0]['workload']
                        task.assign_schedule(target_nid, 0, model_workload)
                        if node.admit_task(task):
                            tasks_admitted += 1
                            arrival_workload_gflops[task.service_id] += task.required_workload_gflops
            
            logger.info(f"   Target Node: {target_nid}")
            logger.info(f"   Tasks Admitted: {tasks_admitted}")
            for sid in node.placed_services:
                q_len = len(node.queues.get(sid, []))
                a_load = arrival_workload_gflops.get(sid, 0.0)
                backlog = node.backlogs.get(sid, 0.0)
                logger.info(f"   [Pre-Process] Svc {sid}: Queue Length: {q_len} | Arrival: {a_load:8.2f} GFLOPS | Total Backlog: {backlog:8.2f} GFLOPS")

            # Xử lý timeslot
            completed, energy = node.process_timeslot(
                current_time_elapsed=0.1, 
                slot_duration=slot_duration, 
                V_param=V_param
            )
            
            # Log kết quả cho từng service đã đặt
            for sid in node.placed_services:
                cpu_alloc = node.last_cpu_allocations.get(sid, 0.0)
                rem_backlog = node.backlogs.get(sid, 0.0)
                logger.info(f"   [Post-Process] Svc {sid}: Alloc: {cpu_alloc:8.2f} GFLOPS | Remaining Backlog: {rem_backlog:8.2f} GFLOPS")
            
            logger.info(f"   Total Energy: {energy:.4f} J | Tasks Finished: {len(completed)}")
            
            # Kiểm tra các ràng buộc
            self.assertGreaterEqual(energy, 0)
            total_alloc = sum(node.last_cpu_allocations.values())
            self.assertLessEqual(total_alloc, node.cpu_capacity + 1e-4) # Cho phép sai số float

if __name__ == '__main__':
    unittest.main()
