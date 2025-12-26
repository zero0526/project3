import sys
import os
import yaml
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.entities.node import ComputingNode
from src.entities.task import Task

def run_test():
    print("=== TESTING PHASE 3: MATHEMATICAL MECHANISMS (KKT & ENERGY) ===\n")

    # 1. Setup Node
    with open("configs/network_params.yaml") as f: net_conf = yaml.safe_load(f)
    with open("configs/services.yaml") as f: svc_conf = yaml.safe_load(f)
    
    node_specs = net_conf['nodes']['edge']
    node = ComputingNode("Test_Node", node_specs)
    
    # Deploy Service 0 (Image Class) & Service 1 (Obj Detect)
    # Service 0: Omega=1. Service 1: Omega=0.
    placement = [1, 1, 0, 0, 0]
    node.update_placement(placement, svc_conf['services'])
    
    print("[1] Initial State:")
    print(f"    Services deployed: {list(node.placed_services.keys())}")
    
    # 2. Inject Load (Tạo áp lực hàng đợi lệch nhau)
    # Service 0: Hàng đợi RẤT LỚN (50 GFLOPS)
    task1 = Task("T1", 0, "U1", 2, 0.5, 0.9, 1, 0)
    task1.required_workload = 50.0 
    node.admit_task(task1)
    
    # Service 1: Hàng đợi NHỎ (5 GFLOPS)
    task2 = Task("T2", 1, "U2", 4, 0.8, 0.8, 0, 0)
    task2.required_workload = 5.0
    node.admit_task(task2)
    
    print(f"    Queue Backlog Svc 0: {node.backlogs[0]} GFLOPS (High Load)")
    print(f"    Queue Backlog Svc 1: {node.backlogs[1]} GFLOPS (Low Load)")
    
    # 3. Run KKT Solver (Process Timeslot)
    print("\n[2] Running KKT Solver (V_param = 1.0)...")
    # V nhỏ -> Ưu tiên xả hàng đợi -> CPU chạy mạnh
    completed, energy = node.process_timeslot(slot_duration=1.0, V_param=1.0)
    
    # Vì logic KKT Solver nằm ẩn trong node, ta không lấy được f_alloc trực tiếp để print 
    # trừ khi sửa code node hoặc debug.
    # Tuy nhiên, ta có thể suy luận qua mức giảm backlog.
    
    processed_0 = 50.0 - node.backlogs[0]
    processed_1 = 5.0 - node.backlogs[1]
    
    print(f"    -> Energy Consumed: {energy:.6f} J")
    print(f"    -> Processed Svc 0: {processed_0:.2f} GFLOPS")
    print(f"    -> Processed Svc 1: {processed_1:.2f} GFLOPS")
    
    if processed_0 > processed_1:
        print("    [OK] SUCCESS: KKT allocated more resources to High Load service.")
    else:
        print("    [FAIL] FAILURE: KKT logic might be wrong.")

    # 4. Test Energy-Saving Mode (V_param Rất Lớn)
    print("\n[3] Testing Energy Saving (V_param = 100000)...")
    # V rất lớn -> Z lớn -> f nhỏ -> Tiết kiệm điện
    
    # Reset load
    node.backlogs[0] = 50.0
    
    completed, energy_saving = node.process_timeslot(slot_duration=1.0, V_param=100000.0)
    processed_saving = 50.0 - node.backlogs[0]
    
    print(f"    -> Energy Consumed: {energy_saving:.6f} J")
    print(f"    -> Processed: {processed_saving:.2f} GFLOPS")
    
    if processed_saving < processed_0:
         print("    [OK] SUCCESS: Node throttled down to save energy (V effect).")
    else:
         print("    [FAIL] FAILURE: V parameter did not reduce resource usage.")

    print("\n[OK] PHASE 3 COMPLETE!")

if __name__ == "__main__":
    run_test()