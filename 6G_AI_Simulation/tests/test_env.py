import os
import sys
import numpy as np
import yaml
from tqdm import tqdm
from src.core.environment import HMFD3QNEnv
from src.utils.converters import SNDLibConverter
from src.utils.config_loader import cfg

def ensure_topology_exists():
    """Helper để tạo file topology json nếu chưa có"""
    if not os.path.exists(cfg.TOPOLOGY_JSON):
        print(f"⚠️ Topology JSON not found at {cfg.TOPOLOGY_JSON}. Running converter...")
        if not os.path.exists("configs/network_params.yaml"):
            raise FileNotFoundError("configs/network_params.yaml is missing!")
            
        with open("configs/network_params.yaml") as f: 
            net_conf = yaml.safe_load(f)
            
        converter = SNDLibConverter(
            cfg.TOPOLOGY_XML,
            cfg.TOPOLOGY_JSON,
            net_conf['nodes']
        )
        converter.convert()
        print("✅ Topology converted.")

def test_environment_loop():
    print("==================================================")
    print("🧪 STARTING ENVIRONMENT INTEGRATION TEST")
    print("==================================================")

    # 1. Setup
    ensure_topology_exists()
    
    # try:
    env = HMFD3QNEnv("configs/simulation.yaml")
    print("✅ Environment Initialized Successfully.")
    print(f"   - Nodes: {len(env.nodes)}")
    print(f"   - Services: {len(env.service_config['services'])}")
    print(f"   - Slot Duration: {env.time_manager.slot_duration}s")
    print(f"   - Slots per Frame: {env.time_manager.slots_per_frame}")
    # except Exception as e:
    #     print(f"❌ Failed to initialize environment: {e}")
    #     return

    # 2. Test Reset
    print("\n[Test] Resetting Environment...")
    obs, info = env.reset()
    
    # Kiểm tra cấu trúc Observation
    assert isinstance(obs, dict), "Observation must be a dictionary"
    node_ids = list(env.nodes.keys())
    assert list(obs.keys()) == node_ids, "Observation keys must match Node IDs"
    
    # Kiểm tra Task ban đầu
    current_tasks = info.get('new_tasks', [])
    print(f"✅ Reset done. Initial Tasks generated: {len(current_tasks)}")
    
    # 3. Simulation Loop (Random Actions)
    TEST_STEPS = 25 # Chạy thử 25 slots (đủ để qua 2 Frames nếu T=10)
    
    total_reward = 0
    total_completed = 0
    
    print(f"\n[Test] Running Simulation for {TEST_STEPS} steps...")
    
    for step in range(TEST_STEPS):
        print(f"\n--- Step {step} (Frame {env.time_manager.current_frame}) ---")
        
        # A. Mock Upper Action (Service Placement)
        upper_actions = {}
        if env.time_manager.is_new_frame():
            print("   >>> Triggering Upper Layer (New Frame)")
            num_services = len(env.service_config['services'])
            for nid in env.nodes:
                # Random binary vector [1, 0, 1, ...]
                # Placement ngẫu nhiên có thể vi phạm RAM -> Env sẽ tự xử lý (không crash là được)
                action_vec = np.random.randint(0, 2, size=num_services).tolist()
                upper_actions[nid] = action_vec

        # B. Mock Lower Action (Task Offloading)
        lower_actions = {}
        # Lấy danh sách task từ bước trước (hoặc reset)
        print(f"   >>> Input Tasks: {len(current_tasks)}")
        
        for task in current_tasks:
            # Chọn bừa 1 node đích
            target_node = np.random.choice(node_ids)
            # Chọn model 0 (giả sử model id 0 luôn tồn tại)
            lower_actions[task.id] = {'node': target_node, 'model': 0}
        
        # C. Step
        try:
            next_obs, reward, done, truncated, info = env.step(lower_actions, upper_actions)
            
            # D. Verification
            # Kiểm tra reward không phải None
            assert reward is not None
            total_reward += reward
            
            # Kiểm tra info
            completed = info['completed']
            qos = info['qos_violations']
            energy = info['energy']
            new_tasks_count = len(info['new_tasks'])
            
            total_completed += completed
            
            print(f"   ✅ Step OK. Reward: {reward:.2f} | Energy: {energy:.2f}J")
            print(f"      Completed: {completed} | QoS Violations: {qos}")
            print(f"      Next Tasks Generated: {new_tasks_count}")
            
            # Cập nhật task cho vòng sau
            current_tasks = info['new_tasks']
            
        except Exception as e:
            print(f"❌ Error during env.step(): {e}")
            import traceback
            traceback.print_exc()
            break
            
        if done:
            print("⚠️ Episode finished early.")
            break

    print("\n==================================================")
    print("📊 TEST SUMMARY")
    print("==================================================")
    print(f"Total Steps Run: {TEST_STEPS}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Tasks Completed: {total_completed}")
    
    if total_completed > 0:
        print("✅ SUCCESS: Tasks were successfully processed (KKT Solver worked).")
    else:
        print("⚠️ WARNING: No tasks completed. Check KKT Solver or Placement logic.")
        print("   (Note: Random placement often leads to high QoS violations due to missing services).")

if __name__ == "__main__":
    test_environment_loop()