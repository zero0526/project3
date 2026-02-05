import numpy as np
import torch
from .base_agent import MFD3QNAgent
from src.utils import cfg

class HMFD3QN_Trainer:
    def __init__(self, env):
        self.env = env
        self.num_services = env.num_services
        
        # 1. Khởi tạo Upper Agents (Dành cho Edge/Network Nodes)
        self.upper_agents = {
            node_id: MFD3QNAgent(
                state_dim=env.upper_state_dim,
                action_dim=env.upper_action_dim,
                mf_dim=env.num_services,
                is_upper=True,
                branch_dims=None
            ) for node_id in env.agent_node_ids
        }
        
        # 2. Khởi tạo Lower Agents (Dành cho toàn bộ Terminals)
        self.lower_agents = {
            term_id: MFD3QNAgent(
                state_dim=env.lower_state_dim,
                action_dim=env.lower_action_dim,
                mf_dim=env.mf_lower_dim,
                is_upper=False,
                branch_dims=None
            ) for term_id in env.terminals.keys()
        }

    def _action_to_placement_vec(self, action_id):
        """Chuyển đổi integer action_id sang vector binary [0, 1, 0...] (Multi-hot bitmask)"""
        # bit 0 là service 0, bit 1 là service 1...
        return [(action_id >> i) & 1 for i in range(self.num_services)]

    def train(self, num_episodes):
        # Thiết lập tham số epsilon-greedy kết hợp Boltzmann
        eps_start = 1.0
        eps_end = 0.0 # Sau 1/2 số ep sẽ chuyển hẳn sang Boltzmann (eps=0)
        decay_period = num_episodes // 2 

        for ep in range(num_episodes):
            # Tính epsilon: Giảm dần về 0 trong 1/2 số episode đầu
            if ep < decay_period:
                eps = eps_start - (eps_start - eps_end) * (ep / decay_period)
            else:
                eps = eps_end

            # Reset Environment
            upper_obs, upper_mf = self.env.reset()
            # Khởi tạo MF_prev (m_hat_{t-1}) bằng quan sát ban đầu
            prev_upper_mf = {nid: upper_mf[nid].copy() for nid in self.env.agent_node_ids}
            prev_lower_mf = {tid: np.zeros(self.env.mf_lower_dim) for tid in self.env.terminals}

            done = False
            while not done:
                # --- A. UPPER LEVEL STEP (Bắt đầu Frame) ---
                placement_actions = {}
                current_upper_mf = {}
                
                upper_action_ids = {}
                for nid in self.env.agent_node_ids:
                    # Select Action (Placement) - Thêm eps vào
                    action_id, mf_pred = self.upper_agents[nid].get_action(
                        upper_obs[nid], prev_upper_mf[nid], temperature=1.0, eps=eps
                    )
                    placement_actions[nid] = self._action_to_placement_vec(action_id)
                    upper_action_ids[nid] = action_id
                    current_upper_mf[nid] = mf_pred
                
                # Apply Placement vào môi trường
                self.env.step_upper(placement_actions)

                # --- B. LOWER LEVEL LOOP (T slots trong 1 Frame) ---
                is_new_frame = False
                while not is_new_frame and not done:
                    lower_obs, lower_mf_obs, lower_mask_obs = self.env._get_lower_obs()
                    
                    terminal_actions = {}
                    for tid, agent in self.lower_agents.items():
                        # Lấy quan sát và thực hiện action scheduling (Có kèm Mask và Eps)
                        if lower_obs[tid] is not None:
                            action_id, _ = agent.get_action(
                                lower_obs[tid], lower_mf_obs[tid], 
                                mask=lower_mask_obs[tid], eps=eps
                            )
                            terminal_actions[tid] = action_id
                        else:
                            terminal_actions[tid] = 0 # Dummy action if no task

                    # Môi trường xử lý slot vật lý
                    next_lower_obs, lower_rewards, done, info = self.env.step_lower(terminal_actions)
                    _, next_lower_mf, _ = self.env._get_lower_obs() # Quan sát MF mới (m_hat_{t})
                    
                    # Lưu vào Replay Buffer của Lower Agents
                    for tid, agent in self.lower_agents.items():
                        if lower_obs[tid] is not None:
                            agent.memory.push(
                                lower_obs[tid], terminal_actions[tid], lower_rewards[tid],
                                next_lower_obs[tid], done, lower_mf_obs[tid], next_lower_mf[tid]
                            )
                        # Train Lower Agent mỗi slot
                        agent.train_step()

                    is_new_frame = info['is_new_frame']

                # --- C. UPPER LEVEL FEEDBACK (Kết thúc Frame) ---
                next_upper_obs, next_upper_mf_obs, upper_reward = self.env.get_upper_feedback()
                
                for nid, agent in self.upper_agents.items():
                    # Lấy lại action_id đã chọn ở đầu frame
                    action_id = upper_action_ids[nid]
                    
                    agent.memory.push(
                        upper_obs[nid], action_id, upper_reward,
                        next_upper_obs[nid], done, upper_mf[nid], next_upper_mf_obs[nid]
                    )
                    # Train Upper Agent mỗi frame
                    agent.train_step()
                    
                # Cập nhật Observation cho frame tiếp theo
                upper_obs = next_upper_obs
                upper_mf = next_upper_mf_obs
                prev_upper_mf = next_upper_mf_obs # Dùng cho m_hat_{t-1}

            print(f"Episode {ep+1}/{num_episodes} completed. Eps: {eps:.2f}, Frame Reward: {upper_reward:.2f}")

    def save_models(self, path="models/"):
        import os
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        for nid, agent in self.upper_agents.items():
            agent.save_model(f"{path}/upper_{nid}.pth")
        for tid, agent in self.lower_agents.items():
            agent.save_model(f"{path}/lower_{tid}.pth")
        print(f"Models saved to {path}")