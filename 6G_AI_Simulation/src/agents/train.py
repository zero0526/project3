import numpy as np
import torch
import os
from tqdm import tqdm
from typing import Dict, List, Any

from src import hp
from src.utils.logger import logger
from src.core.environment import HMFD3QNEnv
from src.agents.lower_agent import LowerAgent
from src.agents.upper_agent import UpperAgent

class HMFD3QNTrainer:
    def __init__(self, env: HMFD3QNEnv, upper_agent: UpperAgent, lower_agent: LowerAgent, device="cpu"):
        """
        Quản lý quy trình huấn luyện HMFD3QN.
        
        Args:
            env: Môi trường Gym (HMFD3QNEnv).
            upper_agent: Agent lớp trên (Service Placement).
            lower_agent: Agent lớp dưới (Task Offloading).
            device: 'cuda' hoặc 'cpu'.
        """
        self.env = env
        self.upper_agent = upper_agent
        self.lower_agent = lower_agent
        self.device = device
        
        # Tạo thư mục checkpoint
        os.makedirs(hp.CHECKPOINT_DIR, exist_ok=True)
        
        # State Tracking cho Mean Field
        # Dict {node_id: action_index}
        self.last_upper_actions = {nid: 0 for nid in env.nodes}
        self.last_lower_actions = {nid: 0 for nid in env.nodes}
        
        # Buffer tạm cho Upper Agent (chờ hết Frame mới có Reward tổng)
        # {node_id: {state, action, mf, pred_mf, cumulative_reward}}
        self.upper_pending_experiences = {}

    def train(self, total_episodes: int):
        """Vòng lặp chính huấn luyện qua các Episodes."""
        print(f"🚀 Start Training on {self.device}...")
        
        for episode in range(total_episodes):
            metrics = self._run_single_episode(episode)
            
            # Logging & Saving
            self._log_episode(episode, metrics)
            self._save_checkpoint(episode)

    def _run_single_episode(self, episode_idx: int) -> Dict[str, float]:
        """Chạy logic cho 1 Episode."""
        obs, info = self.env.reset()
        current_tasks = info.get('new_tasks', [])
        
        total_reward = 0
        total_qos = 0
        upper_loss = 0
        lower_loss = 0
        
        # Reset trackers đầu episode
        self.upper_pending_experiences = {}
        self.last_upper_actions = {nid: 0 for nid in self.env.nodes}
        self.last_lower_actions = {nid: 0 for nid in self.env.nodes}

        pbar = tqdm(range(hp.STEPS_PER_EPISODE), desc=f"Ep {episode_idx+1}", leave=False)
        
        for step in pbar:
            upper_actions_dict = {}
            if self.env.time_manager.is_new_frame():
                upper_actions_dict = self._process_upper_level(obs)

            lower_actions_dict, lower_exps, slot_actions_map = self._process_lower_level(current_tasks)


            next_obs, reward, done, truncated, info = self.env.step(lower_actions_dict, upper_actions_dict)
            next_tasks = info['new_tasks']

            total_reward += reward
            total_qos += info['qos_violations']

            
            # A. Train Lower Agent (Ngay lập tức)
            l_loss_val = self._train_lower_agent(lower_exps, slot_actions_map, reward)
            lower_loss += l_loss_val

            # B. Train Upper Agent (Cuối Frame)
            u_loss_val = self._train_upper_agent(reward, next_obs, step)
            upper_loss += u_loss_val

            # -------------------------------------------------------
            # 5. PREPARE NEXT STEP
            # -------------------------------------------------------
            obs = next_obs
            current_tasks = next_tasks
            
            # Update Mean Field Trackers
            self.last_lower_actions.update(slot_actions_map)
            
            pbar.set_postfix({'Rw': f"{reward:.1f}", 'QoS': info['qos_violations']})
            if done: break
            
        return {
            "reward": total_reward,
            "qos": total_qos,
            "upper_loss": upper_loss,
            "lower_loss": lower_loss
        }


    def _process_upper_level(self, obs: Dict) -> Dict:
        """Logic ra quyết định cho Upper Agent."""
        actions_dict = {}
        current_frame_actions = {}
        
        for nid, node_state in obs.items():
            # Tính Mean Field từ hàng xóm
            mf_in = self._get_neighbors_mean_action(
                nid, self.env.logical_neighbors, self.last_upper_actions, self.upper_agent.action_dim
            )
            
            # Chọn hành động
            state_vec = np.array(node_state, dtype=np.float32)
            act_idx, pred_mf = self.upper_agent.get_action(state_vec, mf_in)
            
            # Decode Action (Index -> Binary Vector)
            num_services = len(self.env.service_config['services'])
            binary_vec = [int(x) for x in format(act_idx, f'0{num_services}b')]
            
            actions_dict[nid] = binary_vec
            current_frame_actions[nid] = act_idx
            
            # Lưu pending experience
            self.upper_pending_experiences[nid] = {
                'state': state_vec,
                'action': act_idx,
                'mf': mf_in,
                'pred_mf': pred_mf,
                'frame_reward': 0.0
            }
            
        # Update tracker
        self.last_upper_actions = current_frame_actions
        return actions_dict

    def _process_lower_level(self, tasks: List):
        """Logic ra quyết định cho Lower Agent."""
        actions_dict = {}
        experiences = []
        slot_actions_map = {} # Để tính Real MF cho training
        
        for task in tasks:
            # State Construction: [Size, 0, Deadline, Acc]
            state_vec = np.array([task.size, 0, task.deadline, task.min_accuracy], dtype=np.float32)
            
            # Mean Field Input (Tại terminal)
            term_id = task.terminal_id
            mf_in = self._get_neighbors_mean_action(
                term_id, self.env.logical_neighbors, self.last_lower_actions, self.lower_agent.action_dim
            )
            
            # Select Action
            target_node_idx, pred_mf = self.lower_agent.get_action(state_vec, mf_in)
            
            # Mapping Index -> Node ID
            node_keys = list(self.env.nodes.keys())
            target_node_id = node_keys[target_node_idx]
            
            actions_dict[task.id] = {'node': target_node_id, 'model': 0}
            slot_actions_map[term_id] = target_node_idx
            
            experiences.append({
                'term_id': term_id,
                'state': state_vec,
                'action': target_node_idx,
                'mf': mf_in,
                'pred_mf': pred_mf
            })
            
        return actions_dict, experiences, slot_actions_map

    def _train_lower_agent(self, experiences, slot_actions_map, reward):
        """Lưu buffer và train Lower Agent."""
        if not experiences: return 0.0
        
        for exp in experiences:
            # Tính Real MF (Target)
            real_mf = self._get_neighbors_mean_action(
                exp['term_id'], self.env.logical_neighbors, slot_actions_map, self.lower_agent.action_dim
            )
            
            self.lower_agent.buffer.push(
                exp['state'], exp['action'], reward,
                exp['state'], True, # Done = True (Task finished decision)
                exp['mf'], real_mf
            )
            
        loss, _ = self.lower_agent.train() or (0, 0)
        return loss

    def _train_upper_agent(self, reward, next_obs, current_step):
        """Tích lũy reward và train Upper Agent khi hết Frame."""
        # 1. Tích lũy reward
        for nid in self.upper_pending_experiences:
            self.upper_pending_experiences[nid]['frame_reward'] += reward
            
        # 2. Kiểm tra hết Frame
        if self.env.time_manager.is_new_frame():
            for nid, exp in self.upper_pending_experiences.items():
                # Real MF (Tại thời điểm ra quyết định - xấp xỉ bằng last_upper_actions hiện tại)
                real_mf = self._get_neighbors_mean_action(
                    nid, self.env.logical_neighbors, self.last_upper_actions, self.upper_agent.action_dim
                )
                
                next_state = np.array(next_obs[nid], dtype=np.float32)
                
                self.upper_agent.buffer.push(
                    exp['state'], exp['action'], exp['frame_reward'],
                    next_state, False,
                    exp['mf'], real_mf
                )
            
            # Clear pending & Train
            self.upper_pending_experiences = {}
            loss, _ = self.upper_agent.train() or (0, 0)
            return loss
            
        return 0.0

    def _get_neighbors_mean_action(self, agent_id, neighbors_map, actions_map, action_dim):
        """Hàm tiện ích tính Mean Field."""
        neighbors = neighbors_map.get(agent_id, [])
        if not neighbors: return np.zeros(action_dim)
        
        sum_actions = np.zeros(action_dim)
        count = 0
        for nid in neighbors:
            if nid in actions_map:
                idx = actions_map[nid]
                one_hot = np.zeros(action_dim)
                one_hot[idx] = 1.0
                sum_actions += one_hot
                count += 1
        
        if count == 0: return np.zeros(action_dim)
        return sum_actions / count

    def _log_episode(self, episode, metrics):
        print(f"End Ep {episode+1}: "
              f"Reward={metrics['reward']:.2f}, "
              f"QoS={metrics['qos']}, "
              f"Loss(L/U)={metrics['lower_loss']:.2f}/{metrics['upper_loss']:.2f}")

    def _save_checkpoint(self, episode):
        if (episode + 1) % 10 == 0:
            torch.save(self.upper_agent.q_eval.state_dict(), f"{hp.CHECKPOINT_DIR}/upper_ep{episode+1}.pt")
            torch.save(self.lower_agent.q_eval.state_dict(), f"{hp.CHECKPOINT_DIR}/lower_ep{episode+1}.pt")