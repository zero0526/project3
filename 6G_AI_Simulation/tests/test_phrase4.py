import numpy as np
import yaml
import os
# Đảm bảo import đúng đường dẫn
from src.core.environment import HMFD3QNEnv
from src.utils.converters import SNDLibConverter

def setup_environment():
    # 1. Đảm bảo Topology tồn tại
    if not os.path.exists("configs/topologies/atlanta_processed.json"):
        print("⚠️ Topology JSON not found. Running converter...")
        with open("configs/network_params.yaml") as f: net_conf = yaml.safe_load(f)
        with open("configs/simulation.yaml") as f: sim_conf = yaml.safe_load(f)
        converter = SNDLibConverter(
            sim_conf['simulation']['paths']['topology_xml'],
            sim_conf['simulation']['paths']['topology_json'],
            net_conf['nodes']
        )
        converter.convert()

    # 2. Khởi tạo Env
    env = HMFD3QNEnv("configs/simulation.yaml")
    return env

def generate_random_upper_action(env):
    """
    Sinh placement ngẫu nhiên cho tất cả các node.
    Action: {node_id: [1, 0, 1, ...]}
    """
    actions = {}
    num_services = len(env.service_config['services'])
    for nid in env.nodes.keys():
        # Random binary vector
        actions[nid] = np.random.randint(0, 2, size=num_services).tolist()
    return actions

def generate_random_lower_action(env, tasks):
    """
    Sinh offloading ngẫu nhiên cho danh sách task.
    Action: {task_id: {'node': target_node, 'model': model_id}}
    """
    actions = {}
    node_ids = list(env.nodes.keys())
    
    for task in tasks:
        # 1. Chọn bừa một node đích
        target = np.random.choice(node_ids)
        
        # 2. Chọn bừa một model (Hợp lệ với service của task)
        svc_profile = env.service_config['services'][task.service_id]
        models = svc_profile['models']
        # Chọn random model index (ví dụ service có 2 model id 0, 1)
        model_idx = np.random.choice([m['id'] for m in models])
        
        actions[task.id] = {
            'node': target,
            'model': model_idx
        }
    return actions

def main():
    print("=== STARTING 6G SIMULATION LOOP (RANDOM AGENT) ===")
    
    # 1. Setup
    env = setup_environment()
    obs, _ = env.reset()
    
    # Biến lưu trữ task chưa được xử lý (pending actions)
    # Trong thực tế, Agent quan sát task ở t, ra quyết định ở t, Env thực thi ở t
    # Ở đây ta giả lập flow: Step trả về task mới -> Loop sau Agent ra quyết định cho task đó
    pending_tasks = [] 
    
    # Chạy thử 50 bước (5 Frames nếu T=10)
    total_steps = 50
    
    for step in range(total_steps):
        # A. Upper Action (Nếu là đầu Frame)
        upper_action = None
        if step % env.slots_per_frame == 0:
            print(f"\n[FRAME START] Generating Upper Actions (Placement)...")
            upper_action = generate_random_upper_action(env)
            
        # B. Lower Action (Cho các task đang chờ từ bước trước)
        lower_action = generate_random_lower_action(env, pending_tasks)
        
        # C. Step Environment
        obs, reward, done, truncated, info = env.step(lower_action, upper_action)
        
        # D. Cập nhật pending tasks cho bước sau
        pending_tasks = info['new_tasks']
        
        # E. Logging
        print(f"Slot {step:02d}: "
              f"Energy={info['energy']:.4f}J | "
              f"QoS Violations={info['qos_violations']} | "
              f"Tasks Done={info['completed']} | "
              f"New Tasks={len(pending_tasks)}")
        
        if done:
            break

    print("\n=== SIMULATION FINISHED SUCCESSFULY ===")
    print(f"Total Energy: {env.total_energy:.2f} J")
    print(f"Total QoS Violations: {env.qos_violations}")

if __name__ == "__main__":
    main()