import torch
import numpy as np
import pandas as pd
import time
import os
import json
from src.core.environment import SixGEnvironment
from src.agents.train import HMFD3QN_Trainer
from src.utils import cfg, SimulationMonitor
from src.utils.logger import setup_logger

# Thiết lập logger chi tiết cho việc test và tính chỉnh hyperparameter
test_logger = setup_logger(name="Agent_Test", log_file="data/logs/agent_test.log", mode='w')

def setup_training_hyperparams(num_episodes):
    """Cấu hình các tham số học tập tối ưu cho số lượng episode cụ thể"""
    params = {}
    params['num_episodes'] = num_episodes
    params['lr_q'] = 1e-4
    params['lr_mf'] = 1e-4
    
    # Khởi tạo tham số annealing nhiệt độ (Boltzmann)
    params['initial_temp'] = 0.5
    params['final_temp'] = 0.05
    params['annealing_steps'] = int(num_episodes * 0.7)
    params['temp_decay'] = (params['initial_temp'] - params['final_temp']) / params['annealing_steps']
    
    # Khởi tạo epsilon (Epsilon-greedy)
    params['initial_eps'] = 0.4
    params['eps_decay_end'] = int(num_episodes * 0.5)
    
    # Cấu hình học tập (Decay)
    params['lr_decay_step'] = max(20, num_episodes // 3)
    params['lr_decay_factor'] = 0.5
    
    return params

def test_and_tune():
    test_logger.info("="*50)
    test_logger.info("BẮT ĐẦU QUÁ TRÌNH TEST VÀ TÍNH CHỈNH AGENT")
    test_logger.info("="*50)

    # 1. Load cấu hình dịch vụ
    from src.utils import cfg
    service_config = cfg.services.get('services', [])
    test_logger.info(f"Đã load {len(service_config)} dịch vụ từ cấu hình.")

    # 2. Khởi tạo môi trường
    env = SixGEnvironment(service_config)
    test_logger.info(f"Khởi tạo môi trường với {len(env.nodes)} nodes và {len(env.terminals)} terminals.")
    
    # 3. Khởi tạo Trainer
    trainer = HMFD3QN_Trainer(env)
    test_logger.info("Khởi tạo HMFD3QN Trainer thành công.")

    # 4. Khởi tạo Monitor cho Dashboard
    monitor = SimulationMonitor()

    # Xóa log cũ để theo dõi đợt huấn luyện mới
    history_file = "data/training_history.csv"
    log_file = "data/logs/agent_test.log"
    if os.path.exists(history_file): os.remove(history_file)
    if os.path.exists(log_file): 
        # File này đã được setup_logger mở với mode='w' ở trên nên không cần/không thể xóa thủ công ở đây trên Windows
        pass
    
    test_logger.info("Đã xóa file history và log cũ để bắt đầu đợt mới.")

    # 4. Cấu hình hyperparams cho 150 Epoch
    hparams = setup_training_hyperparams(num_episodes=150)
    num_episodes = hparams['num_episodes']
    
    # Cập nhật cấu hình vào cfg để các Agent nhận diện
    cfg.neuron_net['LR_Q'] = hparams['lr_q']
    cfg.neuron_net['LR_MF'] = hparams['lr_mf']
    
    test_logger.info(f"Cấu hình Test: Episodes={num_episodes}, Device={cfg.device}")
    test_logger.info(f"Hyperparams: LRs=({hparams['lr_q']}, {hparams['lr_mf']}), Epsilon={hparams['initial_eps']}")
    
    current_temp = hparams['initial_temp']
    best_violation_rate = float('inf')

    # 5. Vòng lặp huấn luyện chính
    for ep in range(num_episodes):
        # LR Decay
        if ep > 0 and ep % hparams['lr_decay_step'] == 0:
            for nid in env.agent_node_ids:
                new_lr = trainer.upper_agents[nid].decay_lr(hparams['lr_decay_factor'])
            for tid in env.terminals:
                new_lr = trainer.lower_agents[tid].decay_lr(hparams['lr_decay_factor'])
            test_logger.info(f"--- LEARNING RATE DECAY: New LR = {new_lr:.6f} ---")

        # Epsilon Decay
        if ep < hparams['eps_decay_end']:
            current_eps = hparams['initial_eps'] * (1 - ep / hparams['eps_decay_end'])
        else:
            current_eps = 0.0

        test_logger.info(f"\n>>> BẮT ĐẦU EPISODE {ep+1}/{num_episodes} | Temp: {current_temp:.3f} | Eps: {current_eps:.3f}")
        start_time = time.time()
        
        # Reset Env
        env.current_episode = ep
        upper_obs, current_upper_mf = env.reset()
        # Khởi tạo prev_mf bằng mf hiện tại (t=0) hoặc zeros
        prev_upper_mf = {nid: current_upper_mf[nid].copy() for nid in env.agent_node_ids}
        
        ep_total_reward = 0
        ep_total_energy = 0
        ep_total_violations = 0
        ep_total_tasks = 0
        ep_total_drift = 0
        ep_q_losses = []
        ep_q_vals = []
        ep_mf_losses = []
        
        frame_idx = 0
        done = False
        
        while not done:
            frame_idx += 1
            # --- UPPER LEVEL ---
            placement_actions = {}
            upper_action_ids = {}
            for nid in env.agent_node_ids:
                # Dùng m_{t-1} để dự đoán m_t và chọn action
                action_id, _ = trainer.upper_agents[nid].get_action(
                    upper_obs[nid], prev_upper_mf[nid], temperature=current_temp, eps=current_eps
                )
                placement_actions[nid] = trainer._action_to_placement_vec(action_id)
                upper_action_ids[nid] = action_id
            
            env.step_upper(placement_actions)

            # --- LOWER LEVEL ---
            is_new_frame = False
            while not is_new_frame and not done:
                # Lấy o_t và m_t (ground truth quan sát từ env)
                lower_obs, lower_mf_obs, lower_mask_obs = env._get_lower_obs()
                
                # Để train MFNet(o_t, m_{t-1}) -> m_t, ta cần lưu m_{t-1}
                # Ở đây ta giả định mf_prev của lower là zero hoặc chính là m_t-1 từ slot trước
                if frame_idx == 1: # Slot đầu tiên của frame
                    prev_lower_mf = {tid: lower_mf_obs[tid].copy() for tid in env.terminals}
                
                terminal_actions = {}
                for tid, agent in trainer.lower_agents.items():
                    if lower_obs[tid] is not None:
                        # get_action dùng m_{t-1} (prev_lower_mf) để dự đoán m_t
                        action_id, _ = agent.get_action(lower_obs[tid], prev_lower_mf[tid], mask=lower_mask_obs[tid], temperature=current_temp, eps=current_eps)
                        terminal_actions[tid] = action_id
                    else:
                        terminal_actions[tid] = 0
 
                # Step Env -> Nhận s_{t+1}, m_{t+1}, reward_t
                next_lower_obs, lower_rewards, done, info = env.step_lower(terminal_actions)
                _, next_lower_mf, _ = env._get_lower_obs()
                
                # Train Lower
                for tid, agent in trainer.lower_agents.items():
                    if lower_obs[tid] is not None:
                        # Lưu tuple: (s_t, a_t, r_t, s_{t+1}, done, m_{t-1}, m_t)
                        # Để MFNet học: s_t, m_{t-1} -> m_t
                        agent.memory.push(
                            lower_obs[tid], terminal_actions[tid], lower_rewards[tid],
                            next_lower_obs[tid], done, prev_lower_mf[tid], next_lower_mf[tid]
                        )
                    ql, mfl, q_v = agent.train_step()
                    if ql > 0: 
                        ep_q_losses.append(ql)
                        ep_q_vals.append(q_v)
                    if mfl > 0: ep_mf_losses.append(mfl)

                prev_lower_mf = lower_mf_obs # Cập nhật m_{t-1} cho slot sau
                ep_total_energy += info['energy']
                ep_total_violations += info['violations']
                ep_total_tasks += info['arrival_tasks']
                ep_total_drift += info.get('F1_tau', 0)
                
                # Log components occasionally (every 20 slots)
                if env.time_manager.current_slot % 20 == 0:
                    test_logger.info(f"Slot {env.time_manager.current_slot:3} | Drift: {info.get('F1_tau', 0):.2e} | Energy: {info['energy']:.2f} | Vio: {info['violations']}")
                
                # Cập nhật Live Dashboard (Mỗi 5 slots để tránh xung đột file trên Windows)
                if env.time_manager.current_slot % 5 == 0 or is_new_frame:
                    monitor.log_step(env, env.time_manager.current_slot, info)
                
                is_new_frame = info['is_new_frame']

            # --- UPPER FEEDBACK ---
            next_upper_obs, next_upper_mf_obs, upper_rewards = env.get_upper_feedback()
            
            # ep_total_reward lưu giá trị trung bình để theo dõi độ hội tụ chung
            avg_upper_reward = np.mean(list(upper_rewards.values())) if upper_rewards else 0
            ep_total_reward += avg_upper_reward
            
            # Train Upper Agents
            for nid, agent in trainer.upper_agents.items():
                reward_nid = upper_rewards.get(nid, 0.0)
                # Lấy lại action_id đã chọn ở đầu frame
                action_id = upper_action_ids[nid]
                # Lưu tuple: (s_t, a_t, r_t, s_{t+1}, done, m_{t-1}, m_t)
                agent.memory.push(
                    upper_obs[nid], action_id, reward_nid,
                    next_upper_obs[nid], done, prev_upper_mf[nid], next_upper_mf_obs[nid]
                )
                ql, mfl, q_val = agent.train_step()
                if ql > 0: 
                    ep_q_losses.append(ql)
                    ep_q_vals.append(q_val)
                if mfl > 0: ep_mf_losses.append(mfl)
                
            upper_obs = next_upper_obs
            prev_upper_mf = current_upper_mf # Cập nhật m_{t-1} cho frame sau
            current_upper_mf = next_upper_mf_obs # Đây là m_t cho frame sau
        
        # Annealing temperature
        current_temp = max(hparams['final_temp'], current_temp - hparams['temp_decay'])

        duration = time.time() - start_time
        violation_rate = (ep_total_violations / ep_total_tasks) if ep_total_tasks > 0 else 1.0
        avg_q = np.mean(ep_q_vals) if ep_q_vals else 0
        
        test_logger.info(f"<<< Kết thúc Episode {ep+1} | Time: {duration:.2f}s | Reward: {ep_total_reward:.2f} | SLA Vio: {violation_rate*100:.2f}% | Q-Val: {avg_q:.2f} | Drift: {ep_total_drift:.2e}")
        
        # --- BEST MODEL CHECKPOINTING ---
        if violation_rate < best_violation_rate:
            best_violation_rate = violation_rate
            checkpoint_dir = f"data/checkpoints/best_sla_{ep+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            test_logger.info(f"*** NEW BEST SLA: {best_violation_rate*100:.2f}%! Saving model to {checkpoint_dir} ***")
            
            for nid in env.agent_node_ids:
                trainer.upper_agents[nid].save_model(f"{checkpoint_dir}/upper_{nid}.pth")
            for tid in env.terminals:
                trainer.lower_agents[tid].save_model(f"{checkpoint_dir}/lower_{tid}.pth")

        # --- LOG TO CSV FOR STREAMLIT ---
        
        history_file = "data/training_history.csv"
        avg_q_loss = np.mean(ep_q_losses) if ep_q_losses else 0
        avg_mf_loss = np.mean(ep_mf_losses) if ep_mf_losses else 0
        
        new_data = {
            "episode": ep + 1,
            "reward": ep_total_reward,
            "energy": ep_total_energy,
            "violation_rate": violation_rate * 100,
            "q_loss": avg_q_loss,
            "mf_loss": avg_mf_loss,
            "temp": current_temp,
            "eps": current_eps
        }
        df = pd.DataFrame([new_data])
        if not os.path.exists(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)

    test_logger.info("\n" + "="*50)
    test_logger.info(f"HOÀN THÀNH HUẤN LUYỆN {num_episodes} EPISODES")
    test_logger.info("="*50)
    
    # Quick Summary to Console
    if 'df' in locals():
        best_qos = 100 - df['violation_rate'].min()
        final_energy = df['energy'].iloc[-1]
        print(f"\n🚀 Training Complete!")
        print(f"📊 Best QoS Observed: {best_qos:.2f}%")
        print(f"⚡ Final Energy Consumption: {final_energy:.2f}J")
        print(f"📈 Check plots in: data/plots/convergence_dual_axis.png")

if __name__ == "__main__":
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/checkpoints", exist_ok=True)
    test_and_tune()
