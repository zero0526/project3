import sys
import os
import yaml
import numpy as np

# Thêm đường dẫn để import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.workload_generator import WorkloadGenerator
from src.entities.node import ComputingNode

def run_test():
    print("=== TESTING PHASE 2: ENTITIES & PHYSICS (MULTI-MODEL) ===\n")
    
    # 1. Load Configs
    print("[1] Loading configurations...")
    try:
        with open("configs/simulation.yaml") as f: sim_conf = yaml.safe_load(f)
        with open("configs/services.yaml") as f: svc_conf = yaml.safe_load(f)
        with open("configs/network_params.yaml") as f: net_conf = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Test Workload Generator (Sinh Requests)
    print("\n[2] Testing Workload Generator...")
    dummy_terminals = ["Term_A", "Term_B"]
    
    # Init Generator
    workload_gen = WorkloadGenerator(
        sim_conf['simulation']['workload'], 
        svc_conf['services'], 
        dummy_terminals
    )
    
    # Sinh task cho Slot 1
    tasks = workload_gen.generate(current_time_slot=1)
    print(f"    -> Generated {len(tasks)} tasks.")
    
    if len(tasks) == 0:
        print("    ⚠️ No tasks generated (Check arrival_rate in config). Exiting test.")
        return

    sample_task = tasks[0]
    print(f"    -> Sample Task ID: {sample_task.id}")
    print(f"       Service ID: {sample_task.service_id}")
    print(f"       User Requirement (Min Acc): {sample_task.min_accuracy:.2f}")

    # 3. Simulate Lower Agent (Model Selection)
    print("\n[3] Simulating Lower Agent (Model Selection)...")
    
    # Lấy thông tin service của task này
    svc_profile = svc_conf['services'][sample_task.service_id]
    available_models = svc_profile['models']
    
    print(f"    -> Service '{svc_profile['name']}' has {len(available_models)} models.")
    
    # Logic Agent: Chọn model nhẹ nhất thỏa mãn Acc >= Min Acc
    valid_models = [m for m in available_models if m['accuracy'] >= sample_task.min_accuracy]
    
    if valid_models:
        best_model = min(valid_models, key=lambda x: x['workload'])
        
        # GÁN MODEL VÀO TASK (Hành động quan trọng nhất của Lower Agent)
        sample_task.selected_model_idx = best_model['id']
        sample_task.required_workload = best_model['workload']
        
        print(f"    ✅ Agent selected model: '{best_model['name']}'")
        print(f"       Acc: {best_model['accuracy']} (Req: {sample_task.min_accuracy:.2f})")
        print(f"       Workload: {sample_task.required_workload} GFLOPS")
    else:
        print("    ❌ QoS Violation: No model satisfies accuracy requirement.")
        sample_task.required_workload = 0 # Đánh dấu là fail

    # 4. Test Computing Node (Service Placement)
    print("\n[4] Testing Computing Node (Service Placement)...")
    node_specs = net_conf['nodes']['edge']
    node = ComputingNode("Edge_Node_1", node_specs)
    
    # Giả lập Upper Agent: Đặt TẤT CẢ dịch vụ (để đảm bảo task được nhận)
    num_services = len(svc_conf['services'])
    placement_vector = [1] * num_services 
    
    print(f"    -> Applying Placement Vector: {placement_vector}")
    node.update_placement(placement_vector, svc_conf['services'])
    
    print(f"    -> Placed Services: {list(node.placed_services.keys())}")
    print(f"    -> RAM Used: {node.used_ram:.2f}/{node.ram_capacity} GB")
    print(f"    -> HDD Used: {node.used_hdd:.2f}/{node.hdd_capacity} GB")

    # 5. Test Admission (Đẩy Task vào Node)
    print("\n[5] Testing Task Admission...")
    
    is_admitted = node.admit_task(sample_task)
    
    if is_admitted:
        print(f"    ✅ Task {sample_task.id} ADMITTED.")
        sid = sample_task.service_id
        current_backlog = node.backlogs[sid]
        print(f"    -> Node Queue Backlog for Service {sid}: {current_backlog} GFLOPS")
    else:
        print(f"    ❌ Task {sample_task.id} REJECTED.")
        print("       (Reason: Either Service not placed OR Model not selected)")

    print("\n✅ PHASE 2 TEST COMPLETE!")

if __name__ == "__main__":
    run_test()