import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from src.utils import cfg

class ReplayBuffer:
    """
    Lưu trữ tuple trải nghiệm bao gồm cả Mean Field.
    Transition: (state, action, reward, next_state, done, mf_current, mf_next)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mf, next_mf):
        # Lưu ý: mf là mean field quan sát được tại t (dùng làm input cho Q-Net tại t)
        # next_mf là mean field quan sát được tại t+1 (dùng cho Target Q-Net)
        self.buffer.append((state, action, reward, next_state, done, mf, next_mf))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mf, next_mf = zip(*batch)

        return (
            torch.FloatTensor(np.array(state)).to(cfg.device),
            torch.LongTensor(np.array(action)).unsqueeze(1).to(cfg.device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(cfg.device),
            torch.FloatTensor(np.array(next_state)).to(cfg.device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(cfg.device),
            torch.FloatTensor(np.array(mf)).to(cfg.device),
            torch.FloatTensor(np.array(next_mf)).to(cfg.device)
        )

    def __len__(self):
        return len(self.buffer)

class MeanFieldNet(nn.Module):
    """
    Mạng xấp xỉ Mean Field (Eq. 43 trong bài báo).
    Input: Trạng thái cục bộ (o) + Mean field bước trước (m_hat_{t-1})
    Output: Mean field dự đoán tại bước hiện tại (m_t)
    """
    def __init__(self, state_dim, action_dim):
        super(MeanFieldNet, self).__init__()
        # Input size = State dim + Mean action dim (vì Mean Field là vector xác suất hành động)
        self.fc1 = nn.Linear(state_dim + action_dim, cfg.neuron_net.get("HIDDEN_MF", 50))
        self.fc2 = nn.Linear(cfg.neuron_net.get("HIDDEN_MF", 50), cfg.neuron_net.get("HIDDEN_MF", 50))
        self.fc3 = nn.Linear(cfg.neuron_net.get("HIDDEN_MF", 50), action_dim) 
        
    def forward(self, state, prev_mean_field):
        x = torch.cat([state, prev_mean_field], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Sử dụng Softmax vì đầu ra là phân phối xác suất của hành động đám đông
        return F.softmax(self.fc3(x), dim=1)

class DuelingQNetwork(nn.Module):
    """
    Mạng Dueling Deep Q-Network (Theo Table III).
    Input: State (s) + Mean Field (m)
    Output: Q-values cho tất cả hành động
    """
    def __init__(self, state_dim, action_dim, mf_dim):
        super(DuelingQNetwork, self).__init__()
        # Input size bao gồm cả Mean Field như một ngữ cảnh (Context)
        self.input_dim = state_dim + mf_dim
        
        # Feature Extraction Layers
        self.fc1 = nn.Linear(self.input_dim, cfg.neuron_net.get("HIDDEN_Q1", 128))
        self.fc2 = nn.Linear(cfg.neuron_net.get("HIDDEN_Q1", 128), cfg.neuron_net.get("HIDDEN_Q2", 64))
        
        # --- Dueling Architecture Separation ---
        
        # 1. Value Stream: V(s, m) -> Đánh giá giá trị của trạng thái
        self.value_stream = nn.Sequential(
            nn.Linear(cfg.neuron_net.get("HIDDEN_Q2", 64), cfg.neuron_net.get("HIDDEN_Q2", 64)),
            nn.ReLU(),
            nn.Linear(cfg.neuron_net.get("HIDDEN_Q2", 64), 1)
        )
        
        # 2. Advantage Stream: A(s, m, a) -> Đánh giá ưu thế của từng hành động
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg.neuron_net.get("HIDDEN_Q2", 64), cfg.neuron_net.get("HIDDEN_Q2", 64)),
            nn.ReLU(),
            nn.Linear(cfg.neuron_net.get("HIDDEN_Q2", 64), action_dim)
        )

    def forward(self, state, mean_field):
        # Ghép state và mean field
        x = torch.cat([state, mean_field], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Kết hợp Value và Advantage (Eq. 46 variant - Dueling aggregation)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class HMFD3QNBaseAgent:
    def __init__(self, device):
        self.device = device
        
    def select_action_boltzmann(self, q_values, mask=None, temperature=1.0):
        """
        Chính sách Boltzmann (Eq. 45): pi(a|s) = exp(Q/zeta) / sum(exp(Q/zeta))
        Áp dụng Mask để loại bỏ các Node không hợp lệ.
        """
        q_vals = q_values.detach().cpu().numpy().flatten()
        
        # 1. Tính toán Exponent
        exp_q = np.exp((q_vals - np.max(q_vals)) / temperature)
        
        # 2. Áp dụng Mask (Ép xác suất về 0 cho hành động bị chặn)
        if mask is not None:
            exp_q = exp_q * mask
            
        # 3. Chuẩn hóa thành phân phối xác suất
        sum_exp = np.sum(exp_q)
        if sum_exp == 0:
            # Fallback: Nếu mask chặn sạch (không nên xảy ra), chọn ngẫu nhiên trong mask gốc
            probs = mask / np.sum(mask) if mask is not None and np.sum(mask) > 0 else np.ones_like(q_vals) / len(q_vals)
        else:
            probs = exp_q / sum_exp
        
        action = np.random.choice(len(q_vals), p=probs)
        return action

    def soft_update(self, local_model, target_model, tau=0.005):
        """Soft update tham số mạng Target (Eq. 49)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MFD3QNAgent(HMFD3QNBaseAgent):
    def __init__(self, state_dim, action_dim, mf_dim, is_upper=False):
        super(MFD3QNAgent, self).__init__(cfg.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mf_dim = mf_dim
        
        # Mạng Q (Dueling)
        # Input: state_dim + mf_dim
        # Output: action_dim
        self.q_eval = DuelingQNetwork(state_dim, action_dim, mf_dim).to(self.device)
        self.q_target = DuelingQNetwork(state_dim, action_dim, mf_dim).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        
        # Mạng Mean Field (Xấp xỉ hành động đám đông)
        self.mf_net = MeanFieldNet(state_dim, mf_dim).to(self.device)
        
        # Optimizer
        lr_q = cfg.neuron_net.get("LR_Q", 1e-4)
        lr_mf = cfg.neuron_net.get("LR_MF", 1e-4)
        self.q_optimizer = optim.Adam(self.q_eval.parameters(), lr=lr_q)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=lr_mf) 
        
        # Buffer
        capacity = cfg.neuron_net.get("BUFFER_SIZE", 100000)
        self.memory = ReplayBuffer(capacity)
        
        self.gamma = cfg.neuron_net.get("GAMMA", 0.99)
        self.tau = cfg.neuron_net.get("TAU", 0.005)
        self.batch_size = cfg.neuron_net.get("BATCH_SIZE", 64)

    def get_action(self, state, prev_mf, mask=None, temperature=1.0):
        """Dự đoán Mean Field hiện tại và chọn hành động"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prev_mf_t = torch.FloatTensor(prev_mf).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. Dự đoán Mean Field hiện tại (Eq. 43)
            current_mf = self.mf_net(state_t, prev_mf_t)
            # 2. Tính Q-values (Eq. 46)
            q_values = self.q_eval(state_t, current_mf)
            
        return self.select_action_boltzmann(q_values, mask=mask, temperature=temperature), current_mf.cpu().numpy().flatten()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
            
        states, actions, rewards, next_states, dones, mfs, next_mfs = self.memory.sample(self.batch_size)

        # --- 1. Update Mean Field Net (Dựa trên MSE giữa MF dự đoán và MF quan sát thực tế) ---
        # Trong thực tế training, mfs trong sample chính là m_hat (quan sát từ môi trường)
        # Chúng ta train MF net để nó dự đoán đúng m_hat dựa trên trạng thái
        predicted_mf = self.mf_net(states, mfs) # Sử dụng t-1 để dự đoán t
        mf_loss = nn.MSELoss()(predicted_mf, mfs)
        
        self.mf_optimizer.zero_grad()
        mf_loss.backward()
        self.mf_optimizer.step()

        # --- 2. Update Q-Network (Double DQN + Dueling) ---
        with torch.no_grad():
            # Double DQN: Dùng eval_net chọn action, target_net tính giá trị
            next_mf_pred = self.mf_net(next_states, next_mfs)
            next_q_eval = self.q_eval(next_states, next_mf_pred)
            next_actions = torch.argmax(next_q_eval, dim=1, keepdim=True)
            
            next_q_target = self.q_target(next_states, next_mf_pred)
            max_next_q = next_q_target.gather(1, next_actions)
            
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        current_q = self.q_eval(states, mfs).gather(1, actions)
        q_loss = nn.MSELoss()(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Soft update target network
        self.soft_update(self.q_eval, self.q_target, self.tau)
        
        return q_loss.item(), mf_loss.item()