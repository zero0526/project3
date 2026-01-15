import unittest
import numpy as np
import os
import sys

# Giả sử cấu trúc folder của bạn
sys.path.append(os.getcwd())

from src.utils import cfg
from src.core.environment import SixGEnvironment

class TestSixGEnvironmentIntegrity(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load service config từ config global
        cls.service_config = cfg.services['services']
        cls.env = SixGEnvironment(service_config=cls.service_config)
        
    def setUp(self):
        self.env.reset()
        # Lấy tất cả các node có tài nguyên tính toán (CPU > 0)
        self.node_ids = sorted([nid for nid, node in self.env.nodes.items() if node.cpu_capacity > 0])
        self.term_ids = sorted(list(self.env.terminals.keys()))
        self.num_services = len(self.service_config)

    def test_01_initialization_structure(self):
        """Kiểm tra khởi tạo các thực thể"""
        print("\n[TEST 1] Structure Check")
        self.assertGreater(len(self.node_ids), 0, "Môi trường phải có ít nhất 1 node có tài nguyên tính toán.")
        self.assertGreater(len(self.env.terminals), 0, "Môi trường phải có terminal.")
        
        print(f"  - Tổng số Computing Nodes: {len(self.node_ids)}")
        for nid in self.node_ids:
            node = self.env.nodes[nid]
            print(f"    * Node {nid}: CPU={node.cpu_capacity} GFLOPS, RAM={node.ram_capacity} GB")
            
            # Kiểm tra thuộc tính cơ bản
            self.assertTrue(hasattr(node, 'cpu_capacity'))
            self.assertTrue(hasattr(node, 'backlogs'))
        
        print(f"  - Terminals: {len(self.term_ids)}")

    def test_02_upper_observation_shape(self):
        """Kiểm tra State và Mean Field của Upper Level (Eq. 34-37)"""
        print("\n[TEST 2] Upper Observation Check")
        obs, mf = self.env._get_upper_obs()
        
        # Theo bài báo: obs = [x_v (placement) + phi (popularity)]
        # Kích thước mong đợi: num_services + num_services = 2 * num_services
        expected_dim = 2 * self.num_services
        
        for nid in self.node_ids:
            self.assertEqual(obs[nid].shape[0], expected_dim, f"Lỗi shape obs tại node {nid}")
            self.assertEqual(mf[nid].shape[0], self.num_services, f"Lỗi shape MF tại node {nid}")
            self.assertFalse(np.isnan(obs[nid]).any(), f"Obs của node {nid} chứa NaN")
        print(f"  - Upper Observation Dim: {expected_dim} (Correct)")

    def test_03_lower_observation_and_mf_logic(self):
        """Kiểm tra State và Mean Field của Lower Level (Eq. 51)"""
        print("\n[TEST 3] Lower Observation & Mean Field Check")
        obs, mf = self.env._get_lower_obs()
        
        # MF vector dim = num_nodes + max_models_in_a_service
        expected_mf_dim = self.env.mf_vector_dim
        
        for tid in self.term_ids:
            if obs[tid] is not None:
                # Kiểm tra obs không rỗng khi có task
                self.assertGreater(obs[tid].shape[0], 0)
            
            self.assertEqual(mf[tid].shape[0], expected_mf_dim, f"Lỗi shape MF Lower tại {tid}")
            self.assertFalse(np.isnan(mf[tid]).any(), f"MF Lower của {tid} chứa NaN")
        print(f"  - Lower Mean Field Dim: {expected_mf_dim} (Correct)")

    def test_04_one_step_execution(self):
        """Chạy thử 1 bước Lower Step và kiểm tra Reward"""
        print("\n[TEST 4] Execution & Reward Sanity Check")
        
        # Tạo dummy actions cho các terminal
        # Action: {terminal_id: (target_node_id, model_index)}
        import random
        dummy_actions = {}
        for tid in self.term_ids:
            # Chọn ngẫu nhiên một node có tài nguyên để gửi task
            target_node = random.choice(self.node_ids)
            dummy_actions[tid] = (target_node, 0) 
            
        next_obs, rewards, done, info = self.env.step_lower(dummy_actions)
        
        # Kiểm tra các key quan trọng trong info
        self.assertIn("energy", info)
        self.assertIn("F1_tau", info)
        self.assertIn("qos_violations", info)
        
        # Reward phải là giá trị số (float/int)
        for tid in self.term_ids:
            self.assertIsInstance(rewards[tid], (float, int, np.float32))
            
        print(f"  - Slot Energy: {info['energy']:.4f}, F1: {info['F1_tau']:.4f}")
        print(f"  - Lower Reward Sample: {rewards[self.term_ids[0]]:.4f}")

    def test_05_timescale_transition(self):
        """Kiểm tra logic chuyển giao giữa Slot (Lower) và Frame (Upper)"""
        print("\n[TEST 5] Two-Time-Scale Logic Check")
        
        T = cfg.task_param.get('T', 10) # Số slot trong 1 frame
        
        # 1. Thực hiện Upper Action (Placement)
        placement_actions = {nid: [1] * self.num_services for nid in self.node_ids}
        self.env.step_upper(placement_actions)
        
        # 2. Chạy hết 1 Frame (T slots)
        for tau in range(T):
            dummy_actions = {tid: (self.node_ids[0], 0) for tid in self.term_ids}
            _, _, _, info = self.env.step_lower(dummy_actions)
            
            if tau < T - 1:
                self.assertFalse(info['is_new_frame'], f"Lỗi: Frame kết thúc sớm tại slot {tau}")
            else:
                self.assertTrue(info['is_new_frame'], "Lỗi: Môi trường không nhận diện kết thúc Frame")
        
        # 3. Kiểm tra Upper Reward sau khi kết thúc frame
        upper_obs, neighbors, upper_reward = self.env.get_upper_reward_and_next_state()
        self.assertIsInstance(upper_reward, (float, int, np.float32))
        self.assertLessEqual(upper_reward, 0, "Upper Reward (Reward = -F1) thường phải <= 0")
        
        print(f"  - Frame completed successfully. Upper Reward: {upper_reward:.4f}")

if __name__ == "__main__":
    unittest.main()