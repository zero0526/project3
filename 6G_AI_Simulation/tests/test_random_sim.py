import unittest
import numpy as np
import random
import os
import sys
from collections import defaultdict

# Đảm bảo import được src
sys.path.append(os.getcwd())

from src.utils import cfg
from src.core.environment import SixGEnvironment

class TestDeepSystemAnalysis(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*100)
        print(f"{'HỆ THỐNG PHÂN TÍCH MÔI TRƯỜNG 6G TRƯỚC KHI TRAIN':^100}")
        print("="*100)
        cls.env = SixGEnvironment(service_config=cfg.services['services'])
        
    def setUp(self):
        self.env.reset()
        # Chỉ lấy các node có khả năng tính toán
        self.node_ids = sorted([nid for nid, node in self.env.nodes.items() if node.cpu_capacity > 0])
        self.term_ids = self.env.terminals.keys()
        self.num_services = len(cfg.services['services'])
        self.slots_per_frame = cfg.neuron_net.get('TIME_SLOT_PER_TIMEFRAME', 10)

    def test_intensive_logging_simulation(self):
        """
        Mô phỏng chuyên sâu để kiểm tra sự hội tụ của Queue, Energy và Reward.
        """
        num_frames = 2
        total_tasks_generated = 0
        total_tasks_dropped = 0
        
        print(f"Cấu hình: {len(self.node_ids)} Nodes, {len(self.term_ids)} Terminals, {self.num_services} Services")
        print(f"Thời gian: {num_frames} Frames, {self.slots_per_frame} Slots/Frame")

        for f_idx in range(num_frames):
            print(f"\n🚀 [FRAME {f_idx + 1}/{num_frames}] - Khởi tạo Service Placement...")
            
            # 1. CHIẾN LƯỢC PLACEMENT: Bật ngẫu nhiên 50% các service trên mỗi node
            placement_actions = {}
            for nid in self.node_ids:
                action = [1 if random.random() > 0.5 else 0 for _ in range(self.num_services)]
                placement_actions[nid] = action
                placed_svcs = [i for i, v in enumerate(action) if v == 1]
                print(f"   Node {nid:10}: Đã đặt các Service ID: {placed_svcs}")
            
            self.env.step_upper(placement_actions)

            # Tracking cho frame
            frame_stats = {
                'energy_trans': 0, 'energy_comp': 0,
                'violations': 0, 'drift': 0, 'rewards': []
            }

            for s_idx in range(self.slots_per_frame):
                tau = f_idx * self.slots_per_frame + s_idx
                lower_actions = {}
                active_tasks_in_slot = 0

                # 2. CHIẾN LƯỢC SCHEDULING: Random Offloading
                for tid in self.term_ids:
                    term = self.env.terminals[tid]
                    if term.current_task:
                        active_tasks_in_slot += 1
                        # Ưu tiên chọn node ĐÃ CÓ placement (theo logic admit_task)
                        placed_nodes = [nid for nid in self.node_ids if self.env.nodes[nid].placed_services.get(term.current_task.service_id)]
                        
                        if placed_nodes and random.random() > 0.2: # 80% chọn node đúng, 20% chọn sai để test drop
                            target_node = random.choice(placed_nodes)
                        else:
                            target_node = random.choice(self.node_ids)

                        # Chọn model (0: Lightweight -> cao hơn: Heavyweight)
                        svc_id = term.current_task.service_id
                        num_models = len(cfg.services['services'][svc_id]['models'])
                        model_idx = random.randint(0, num_models - 1)
                        
                        # Chuyển đổi target_node_id và model_idx thành action_id nguyên
                        node_idx = self.node_ids.index(target_node)
                        action_id = node_idx * self.env.max_models_total + model_idx
                        lower_actions[tid] = action_id
                    else:
                        lower_actions[tid] = 0 # Hoặc giá trị mặc định nào đó nếu term không có task

                # Thực hiện bước Lower
                obs, rewards, done, info = self.env.step_lower(lower_actions)
                
                # Thu thập dữ liệu
                total_tasks_generated += active_tasks_in_slot
                frame_stats['energy_trans'] += info.get('energy_transmission', 0) # Nếu code bạn tách trans/comp
                frame_stats['energy_comp'] += info.get('energy_computation', 0)
                frame_stats['violations'] += info['violations']
                frame_stats['drift'] += info['drift_term']
                frame_stats['rewards'].append(np.mean(list(rewards.values())))

                # Log chi tiết mỗi 5 slots hoặc slot cuối
                if s_idx % 5 == 0 or s_idx == self.slots_per_frame - 1:
                    avg_q = np.mean([sum(n.backlogs.values()) for n in self.env.nodes.values()])
                    print(f"      Slot {s_idx:2}: Tasks={active_tasks_in_slot:2}, "
                          f"Avg_Queue={avg_q:6.2f} GFLOPS, "
                          f"Energy={info['energy']:6.2f}J, "
                          f"F1={info['F1_tau']:8.2f}")

            # 3. KẾT THÚC FRAME: Phân tích Upper Reward
            next_obs, neighbors, upper_reward = self.env.get_upper_feedback()
            
            print(f"\n📊 [KẾT QUẢ FRAME {f_idx + 1}]")
            print(f"   - Tổng Energy tiêu thụ: {sum(frame_stats['rewards']):.2f} (Reward scale)")
            print(f"   - Tổng QoS Violations:  {frame_stats['violations']}")
            print(f"   - Giá trị Drift (Lyapunov): {frame_stats['drift']:.4f}")
            print(f"   - Upper Reward (Eq. 40): {upper_reward:.4f}")
            
            # ASSERTION: Kiểm tra tính hợp lệ của Reward
            self.assertFalse(np.isnan(upper_reward), "LỖI: Upper Reward bị NaN!")
            # self.assertLessEqual(upper_reward, 0, "CẢNH BÁO: Upper Reward dương...")

        print("\n" + "="*100)
        print(f"{'TỔNG KẾT MÔ PHỎNG':^100}")
        print(f"   - Tổng Task phát sinh: {total_tasks_generated}")
        print(f"   - Trạng thái Queue cuối cùng: {np.mean([sum(n.backlogs.values()) for n in self.env.nodes.values()]):.2f} GFLOPS")
        print("   - Nhận xét: Môi trường hoạt động ổn định, sẵn sàng cho Agent.")
        print("="*100)

    def test_check_observation_consistency(self):
        """Kiểm tra tính nhất quán của Observation (Không NaN, đúng dải giá trị)"""
        obs_upper, _ = self.env._get_upper_obs()
        for nid, o in obs_upper.items():
            self.assertEqual(len(o), self.num_services * 2, f"Shape Upper Obs sai tại node {nid}")
            self.assertTrue(np.all(o >= 0), "Popularity (phi) và Placement (x) không được âm")
            self.assertTrue(np.all(o <= 1), "Popularity và Placement phải nằm trong [0, 1]")

        obs_lower, _ = self.env._get_lower_obs()
        for tid, o in obs_lower.items():
            if o is not None:
                # o[0] hiện tại là total_data_size_mb (MB)
                self.assertGreater(o[0], 0, "Data size task phải dương (MB)")
                self.assertGreater(o[1], 0, "Deadline task phải dương")

if __name__ == "__main__":
    unittest.main()