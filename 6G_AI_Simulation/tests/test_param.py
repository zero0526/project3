import unittest
import numpy as np
from collections import defaultdict

# Import hệ thống
from src.utils import cfg
from src.core.environment import SixGEnvironment

class TestDebugScale(unittest.TestCase):
    
    def setUp(self):
        print("\n" + "="*60)
        print("DEBUG SETUP: Phân tích nguyên nhân bùng nổ giá trị")
        self.env = SixGEnvironment(service_config=cfg.services['services'])
        self.env.reset()
        print(cfg.lypa_coef)
        # Lấy thông số cơ bản
        self.slot_duration = self.env.time_manager.slot_duration
        self.node_ids = sorted(list(self.env.nodes.keys()))
        self.term_ids = sorted(list(self.env.terminals.keys()))

    def test_01_compare_capacity_vs_workload(self):
        """
        KIỂM TRA 1: So sánh Năng lực xử lý (Node CPU) vs Tải đến (Task Workload)
        Nguyên nhân phổ biến: Arrival Rate quá cao hoặc Workload > CPU Capacity.
        """
        print("\n[PHÂN TÍCH 1] Node Capacity vs Task Workload")
        
        # 1. Lấy năng lực CPU trung bình của Node
        sample_node = self.env.nodes[self.node_ids[0]]
        cpu_cap = sample_node.cpu_capacity
        print(f"  -> Node CPU Capacity (f_max): {cpu_cap:,.2f} GFLOPS (hoặc đơn vị gốc)")
        
        # 2. Lấy Workload của 1 Task trung bình
        print("  -> Task Workload per Service:")
        total_avg_workload = 0
        for svc in cfg.services['services']:
            # Lấy model đầu tiên làm mẫu
            model = svc['models'][0]
            batch_size = cfg.task_param['default_batch_size']
            task_load = model['workload'] * batch_size
            total_avg_workload += task_load
            print(f"     - Service {svc['id']} ({svc['name']}): {task_load:,.2f} (per task)")

        avg_task_load = total_avg_workload / len(cfg.services['services'])
        
        # 3. Tính tổng tải ước lượng đè lên 1 Node trong 1 slot
        # Giả sử Arrival Rate = 1.0, có N terminals, M nodes.
        # Trung bình 1 Node gánh: (N / M) terminals
        num_terms = len(self.env.terminals)
        num_nodes = len(self.env.nodes)
        terms_per_node = num_terms / num_nodes
        arrival_rate = cfg.task_param['arrival_rate']
        
        estimated_arrival_load = terms_per_node * arrival_rate * avg_task_load
        
        print(f"\n  [ĐÁNH GIÁ CÂN BẰNG TẢI]")
        print(f"  - Số terminal trung bình mỗi node gánh: {terms_per_node:.1f}")
        print(f"  - Tổng Workload ước tính đến 1 node/slot: {estimated_arrival_load:,.2f}")
        print(f"  - Khả năng xử lý tối đa của node/slot:    {cpu_cap * self.slot_duration:,.2f}")
        
        ratio = estimated_arrival_load / (cpu_cap * self.slot_duration)
        print(f"  -> TỶ LỆ TẢI/NĂNG LỰC (Load Ratio): {ratio:.2f}")
        
        if ratio > 1.0:
            print("  >>> CẢNH BÁO ĐỎ: Hệ thống quá tải (Overloaded) ngay từ đầu!")
            print("      Hàng đợi sẽ tăng vô hạn -> Drift bùng nổ là điều chắc chắn.")
        else:
            print("  >>> Hệ thống có vẻ ổn định về mặt công suất (Underloaded).")

    def test_02_breakdown_drift_calculation(self):
        """
        KIỂM TRA 2: Mổ xẻ công thức Drift trong 1 Slot cụ thể.
        Drift Component = Q * (A - W)
        Chúng ta sẽ xem Q, A, W lớn đến mức nào.
        """
        print("\n[PHÂN TÍCH 2] Chi tiết thành phần Drift (Q, A, W)")
        
        # Chạy 1 vài slot để tích lũy hàng đợi
        print("  -> Chạy warm-up 5 slots...")
        for _ in range(5):
            actions = {tid: (self.node_ids[0], 0) for tid in self.term_ids} # Dồn hết vào Node 0
            self.env.step_lower(actions)
            
        # Bắt đầu phân tích tại slot 6
        target_node = self.env.nodes[self.node_ids[0]]
        
        # Giả lập input cho slot tiếp theo
        # Dồn 5 task vào node này
        term_subset = self.term_ids[:5]
        actions = {tid: (target_node.id, 0) for tid in self.term_ids} # Dồn toàn bộ terminal vào
        
        # Hook để lấy dữ liệu trước khi step_lower tính toán xong
        # (Ở đây ta chạy step_lower và phân tích kết quả trả về)
        next_obs, rewards, done, info = self.env.step_lower(actions)
        
        print(f"\n  [KẾT QUẢ SLOT VỪA CHẠY]")
        print(f"  -> Total Drift Term: {info['drift_term']:,.2f}")
        print(f"  -> Total Energy:     {info['energy']:,.2f}")
        
        print("\n  [CHI TIẾT TỪNG SERVICE TẠI NODE 0]")
        # Truy cập trực tiếp vào biến nội bộ của Node để xem
        total_drift_manual = 0
        
        for svc_id in range(self.env.num_services):
            # Q: Backlog hiện tại
            Q = target_node.backlogs.get(svc_id, 0.0)
            
            # W: Đã xử lý (Lấy từ allocated f)
            f_alloc = target_node.last_cpu_allocations.get(svc_id, 0.0)
            W = f_alloc * self.slot_duration
            
            # A: Ước lượng (Vì biến A cục bộ trong hàm step_lower không lấy ra được, ta nhìn vào Q change)
            # Q_new ~ Q_old - W + A => A ~ Q_new - Q_old + W. 
            # Tuy nhiên để đơn giản, ta nhìn vào độ lớn của Q và W.
            
            print(f"  --- Service {svc_id} ---")
            print(f"      + Queue Backlog (Q): {Q:,.2f}")
            print(f"      + Processed (W):     {W:,.2f}")
            print(f"      + CPU Alloc (f):     {f_alloc:,.2f}")
            
            # Phân tích mức độ đóng góp vào Drift
            # Drift ~= Q * A (nếu W nhỏ) hoặc Q * (A-W)
            # Nếu Q ~ 1000 và A ~ 100 -> Drift ~ 100,000 cho 1 service/1 node
            
            if Q > 1000:
                print(f"      >>> CẢNH BÁO: Queue quá lớn ({Q:,.0f}). Bình phương hoặc nhân với A sẽ ra số khổng lồ.")

    def test_03_simulate_reward_calculation(self):
        """
        KIỂM TRA 3: Mô phỏng tính toán Reward Upper
        """
        print("\n[PHÂN TÍCH 3] Mô phỏng tính Reward Upper")
        
        # Lấy Drift từ test 2
        sample_drift = 5e8 # Giả sử 500 triệu (như log của bạn)
        sample_energy = 30.0
        V_param = cfg.lypa_coef
        
        print(f"  Giả định:")
        print(f"  - Accumulated Drift (Frame): {sample_drift:,.0f}")
        print(f"  - Accumulated Energy:        {sample_energy:,.0f}")
        print(f"  - Tham số V (lypa_coef):     {V_param}")
        
        # F1 = Drift + V * Energy
        term_1 = sample_drift
        term_2 = V_param * sample_energy
        
        print(f"\n  [CÔNG THỨC F1]")
        print(f"  F1 = {term_1:,.0f} (Drift) + {term_2:,.4f} (V*Energy)")
        
        total_F1 = term_1 + term_2
        print(f"  -> Total F1: {total_F1:,.0f}")
        print(f"  -> Upper Reward (-F1): {-total_F1:,.0f}")
        
        print("\n  [KẾT LUẬN]")
        if term_1 > term_2 * 1000:
            print("  >>> Drift đang ÁP ĐẢO hoàn toàn Energy (Gấp >1000 lần).")
            print("  >>> Cần TĂNG V hoặc SCALE DRIFT xuống.")
        
        if abs(total_F1) > 1e5:
            print("  >>> Giá trị Reward tuyệt đối quá lớn (>100,000). Neural Network sẽ không học được.")
            print("  >>> Cần SCALE REWARD (ví dụ: nhân với 1e-9).")

if __name__ == '__main__':
    unittest.main()