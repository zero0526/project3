import unittest
import numpy as np
import os
import sys

# Đảm bảo import được src
sys.path.append(os.getcwd())

from src.utils import cfg
from src.core.environment import SixGEnvironment
from src.entities.task import Task

class TestKKTInternalAllocation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*80)
        print("BẮT ĐẦU CHẨN ĐOÁN CHI TIẾT PHÂN BỔ TÀI NGUYÊN (KKT DIAGNOSTIC)")
        print("="*80)
        # Sử dụng config thực tế
        cls.env = SixGEnvironment(service_config=cfg.services['services'])
        
    def test_allocation_lifecycle(self):
        """
        Kiểm tra vòng đời phân bổ: Task -> Admit -> KKT Params -> f_alloc -> Process
        """
        # 1. Chuẩn bị Node và Service
        node_id = list(self.env.nodes.keys())[0]
        node = self.env.nodes[node_id]
        target_svc_id = 0
        slot_duration = self.env.time_manager.slot_duration # 0.01s (mặc định)
        
        print(f"\n[1] SETUP: Node={node_id}, CPU_Cap={node.cpu_capacity} GFLOPS, Slot={slot_duration}s")
        
        # Đặt service lên node
        action_vec = [0] * self.env.num_services
        action_vec[target_svc_id] = 1
        self.env.step_upper({node_id: action_vec})
        
        # 2. Tạo Task có Workload lớn và Deadline gấp
        # Giả sử Node có 3000 GFLOPS/s -> 1 slot xử lý được 30 GFLOPS.
        # Ta tạo 3 tasks, mỗi task 50 GFLOPS (Tổng 150 GFLOPS). 
        # Hệ thống PHẢI mất ít nhất 5 slots để xả hết.
        tasks = []
        for i in range(3):
            # Tạo Task thật
            task = Task(
                task_id=f"debug_task_{i}",
                terminal_id="term_0",
                source_node_id="N0",
                service_id=target_svc_id,
                batch_size=20,
                deadline=5, # Rất gấp (chỉ 5 slots)
                min_accuracy=0.8,
                created_at=0,
                service_info=cfg.services['services'][target_svc_id]
            )
            # Gán workload (Model index 0)
            model_workload = cfg.services['services'][target_svc_id]['models'][0]['workload']
            task.assign_schedule(node_id, 0, model_workload)
            tasks.append(task)
        
        print(f"[2] ADMITTING TASKS: 3 tasks x {tasks[0].required_workload_gflops:.2f} GFLOPS")
        for t in tasks:
            success = node.admit_task(t)
            self.assertTrue(success, f"Node từ chối nhận task {t.id}!")

        current_backlog = node.backlogs[target_svc_id]
        print(f"    -> Hàng đợi hiện tại: {current_backlog:.2f} GFLOPS")

        # 3. CHẠY THỬ 1 SLOT VÀ SOI KKT SOLVER
        print("\n[3] PROCESSING SLOT 1 - PHÂN TÍCH KKT SOLVER")
        
        # Ta sẽ can thiệp vào _compute_optimal_resources để xem các tham số đầu vào của Solver
        active_svcs = [target_svc_id]
        f_alloc_vec, cold_times = node._compute_optimal_resources(
            active_svcs, current_time_elapsed=0.01, slot_duration=slot_duration, V_param=cfg.lypa_coef
        )
        
        f_val = f_alloc_vec[0]
        # In các tham số KKT thực tế
        print(f"    -> KKT Output (f): {f_val:.4f} GFLOPS")
        print(f"    -> Tỉ lệ CPU sử dụng: {(f_val/node.cpu_capacity)*100:.2f}%")
        
        if f_val < 1e-3:
            print("    [!] CẢNH BÁO: Solver cấp CPU gần bằng 0.")
            print(f"    - V_param hiện tại: {cfg.lypa_coef}")
            print(f"    - Kiểm tra lại hàm Solver.solve() hoặc trọng số V.")

        # 4. THỰC THI SLOT
        completed, energy = node.process_timeslot(0.01, slot_duration)
        new_backlog = node.backlogs[target_svc_id]
        
        print(f"\n[4] KẾT QUẢ SAU SLOT:")
        print(f"    - Tasks hoàn thành: {len(completed)}")
        print(f"    - Năng lượng tiêu thụ: {energy:.6f} J")
        print(f"    - Hàng đợi còn lại: {new_backlog:.2f} GFLOPS")
        
        # Nếu hàng đợi không giảm, chứng tỏ f_alloc quá thấp
        if new_backlog >= current_backlog and f_val > 0:
             print("    [!] LỖI: Backlog không giảm dù f_val > 0. Kiểm tra QueueDynamics.update_backlog")

    def test_deadline_enforcement(self):
        """Kiểm tra xem f_min có thực sự tăng khi task sắp hết hạn không"""
        print("\n" + "-"*40)
        print("KIỂM TRA ÉP DEADLINE (f_min Enforcement)")
        node_id = list(self.env.nodes.keys())[0]
        node = self.env.nodes[node_id]
        target_svc_id = 0
        
        # Xóa sạch queues cũ
        node.reset()
        node.placed_services[target_svc_id] = True
        node.queues[target_svc_id] = []
        node.backlogs[target_svc_id] = 0.0
        
        # Tạo 1 task RẤT GẤP (sắp hết hạn ngay bây giờ)
        # Giả sử slot_duration=0.01, current_time=0.1
        # Task sinh từ lúc 0.0, deadline=0.105 (còn 0.005s - nhỏ hơn 1 slot)
        critical_task = Task(
            task_id="critical_1", 
            terminal_id="term_0", 
            source_node_id="N0", 
            service_id=target_svc_id, 
            batch_size=20, 
            deadline=0.105, 
            min_accuracy=0.8,
            created_at=0, 
            service_info=cfg.services['services'][target_svc_id]
        )
        critical_task.assign_schedule(node_id, 0, 100.0) # Cần 100 GFLOPS
        node.admit_task(critical_task)
        
        # Ở thời điểm 0.1, task này chỉ còn 0.005s. Nó BẮT BUỘC phải xử lý với f_max.
        f_alloc, _ = node._compute_optimal_resources([target_svc_id], 0.1, 0.01, cfg.lypa_coef)
        
        print(f"    - Task cần 100 GFLOPS trong 0.005s còn lại.")
        print(f"    - Solver Allocated f: {f_alloc[0]:.2f} GFLOPS")
        print(f"    - Node f_max: {node.cpu_capacity:.2f} GFLOPS")
        
        self.assertAlmostEqual(f_alloc[0], node.cpu_capacity, delta=10.0, 
                               msg="Lỗi! Khi task sắp hết hạn, f_alloc phải xấp xỉ f_max.")

if __name__ == "__main__":
    unittest.main()
