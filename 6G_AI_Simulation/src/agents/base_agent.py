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
        hid = cfg.neuron_net.get("HIDDEN_MF", 50)
        # Input size = State dim + Mean action dim
        self.fc1 = nn.Linear(state_dim + action_dim, hid)
        self.ln1 = nn.LayerNorm(hid)
        self.fc2 = nn.Linear(hid, hid)
        self.ln2 = nn.LayerNorm(hid)
        self.fc3 = nn.Linear(hid, action_dim) 
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Chuyển từ Xavier sang He (Kaiming) Normal
                # nonlinearity='relu' giúp PyTorch tính toán độ lệch chuẩn (std) tối ưu cho ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Giữ nguyên phần khởi tạo lớp output siêu nhỏ để đảm bảo tính ổn định ban đầu
        # Phần này không nên dùng He vì nó là lớp cuối (Softmax/Linear), không đi qua ReLU nữa
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, state, prev_mean_field):
        x = torch.cat([state, prev_mean_field], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
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
        hid1 = cfg.neuron_net.get("HIDDEN_Q1", 128)
        hid2 = cfg.neuron_net.get("HIDDEN_Q2", 64)
        
        # Feature Extraction Layers
        self.fc1 = nn.Linear(self.input_dim, hid1)
        self.ln1 = nn.LayerNorm(hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.ln2 = nn.LayerNorm(hid2)
        
        # --- Dueling Architecture Separation ---
        # 1. Value Stream: V(s, m)
        self.value_stream = nn.Sequential(
            nn.Linear(hid2, hid2),
            nn.LayerNorm(hid2),
            nn.ReLU(),
            nn.Linear(hid2, 1)
        )
        
        # 2. Advantage Stream: A(s, m, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hid2, hid2),
            nn.LayerNorm(hid2),
            nn.ReLU(),
            nn.Linear(hid2, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize Advantage and Value output layers with small weights
        # This keeps initial Q-values close to zero, aiding stability
        nn.init.uniform_(self.value_stream[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(self.value_stream[-1].bias, 0)
        nn.init.uniform_(self.advantage_stream[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(self.advantage_stream[-1].bias, 0)

    def forward(self, state, mean_field):
        # Ghép state và mean field
        x = torch.cat([state, mean_field], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Kết hợp Value và Advantage (Eq. 46 variant - Dueling aggregation)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class HMFD3QNBaseAgent:
    def __init__(self, device):
        self.device = device
        
    def select_action_boltzmann(self, q_values, mask=None, temperature=1.0, eps=0.0, branch_dims=None):
        """
        Chính sách Boltzmann kết hợp Epsilon-greedy cho một hoặc nhiều nhánh hành động.
        branch_dims: List các số lượng hành động cho mỗi nhánh (ví dụ: [num_nodes, num_models]).
        """
        q_vals_all = q_values.detach().cpu().numpy().flatten()
        
        if branch_dims is None:
            branch_dims = [len(q_vals_all)]
            
        actions = []
        start_idx = 0
        for b_dim in branch_dims:
            b_q_vals = q_vals_all[start_idx : start_idx + b_dim]
            b_mask = None
            if mask is not None:
                # Nếu là branching, mask cũng phải được split hoặc truyền đúng định dạng.
                # Ở đây giả định mask đơn giản cho từng nhánh nếu cần.
                # Tuy nhiên, trong 6G Task Scheduling, Mask thường phức tạp (Node x Model).
                # Ta sẽ xử lý mask riêng ở level cao hơn nếu có branching.
                if len(mask) == b_dim:
                    b_mask = mask
                elif len(mask) == len(q_vals_all):
                    b_mask = mask[start_idx : start_idx + b_dim]
            
            # Epsilon-greedy exploration
            if random.random() < eps:
                if b_mask is not None and np.sum(b_mask) > 0:
                    p_rand = b_mask / np.sum(b_mask)
                else:
                    p_rand = np.ones(b_dim) / b_dim
                actions.append(np.random.choice(b_dim, p=p_rand))
            else:
                # Boltzmann exploitation
                exp_q = np.exp((b_q_vals - np.max(b_q_vals)) / temperature)
                if b_mask is not None:
                    exp_q = exp_q * b_mask
                
                sum_exp = np.sum(exp_q)
                if sum_exp <= 0 or np.isnan(sum_exp):
                    if b_mask is not None and np.sum(b_mask) > 0:
                        probs = b_mask / np.sum(b_mask)
                    else:
                        probs = np.ones(b_dim) / b_dim
                else:
                    probs = exp_q / sum_exp
                
                actions.append(np.random.choice(b_dim, p=probs))
            
            start_idx += b_dim

        return actions[0] if len(actions) == 1 else tuple(actions)

    def soft_update(self, local_model, target_model, tau=0.005):
        """Soft update tham số mạng Target (Eq. 49)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MFD3QNAgent(HMFD3QNBaseAgent):
    def __init__(self, state_dim, action_dim, mf_dim, is_upper=False, branch_dims=None):
        super(MFD3QNAgent, self).__init__(cfg.device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mf_dim = mf_dim
        self.branch_dims = branch_dims # Ví dụ: [num_nodes, num_models]
        
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
        capacity = int(cfg.neuron_net.get("BUFFER_SIZE", 300000))
        self.memory = ReplayBuffer(capacity)
        
        self.gamma = cfg.neuron_net.get("GAMMA", 0.99)
        self.tau = cfg.neuron_net.get("TAU", 0.005)
        self.batch_size = cfg.neuron_net.get("BATCH_SIZE", 64)

    def get_action(self, state, prev_mf, mask=None, temperature=1.0, eps=0.0):
        """Dự đoán Mean Field hiện tại và chọn hành động"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prev_mf_t = torch.FloatTensor(prev_mf).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            current_mf = self.mf_net(state_t, prev_mf_t)
            q_values = self.q_eval(state_t, current_mf)
            
        # Xử lý mask cho branching
        processed_mask = mask
        if self.branch_dims and mask is not None and len(mask) != self.action_dim:
            # Nếu mask truyền vào là N*M nhưng action_dim là N+M
            # Ta cần nén mask về dạng có thể dùng cho select_action
            # Ở đây đơn giản nhất là select_action tự xử lý hoặc mask được split trước.
            pass

        return self.select_action_boltzmann(q_values, mask=processed_mask, temperature=temperature, eps=eps, branch_dims=self.branch_dims), current_mf.cpu().numpy().flatten()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
            
        states, actions, rewards, next_states, dones, mfs, next_mfs = self.memory.sample(self.batch_size)

        # --- 1. Update Mean Field Net (Dựa trên MSE giữa MF dự đoán và MF quan sát thực tế) ---
        # Trong thực tế training, mfs trong sample chính là m_hat (quan sát từ môi trường)
        # Chúng ta train MF net để nó dự đoán đúng m_hat dựa trên trạng thái
        predicted_mf = self.mf_net(states, mfs) # m_hat_t = MF(s_t, m_{t-1})
        mf_loss = nn.MSELoss()(predicted_mf, next_mfs) # Target là m_t thực tế
        
        self.mf_optimizer.zero_grad()
        mf_loss.backward()
        self.mf_optimizer.step()

        # --- 2. Update Q-Network (Double DQN + Dueling + Branching) ---
        with torch.no_grad():
            next_mf_pred = self.mf_net(next_states, next_mfs)
            next_q_eval = self.q_eval(next_states, next_mf_pred)
            next_q_target = self.q_target(next_states, next_mf_pred)
            
            if self.branch_dims:
                # Tính Target Q cho từng nhánh và lấy trung bình
                target_q_list = []
                start_idx = 0
                for i, b_dim in enumerate(self.branch_dims):
                    # Double DQN: eval chọn action, target tính value
                    b_next_q_eval = next_q_eval[:, start_idx : start_idx + b_dim]
                    b_next_actions = torch.argmax(b_next_q_eval, dim=1, keepdim=True)
                    
                    b_next_q_target = next_q_target[:, start_idx : start_idx + b_dim]
                    b_max_next_q = b_next_q_target.gather(1, b_next_actions)
                    
                    b_target_q = rewards + (1 - dones) * self.gamma * b_max_next_q
                    target_q_list.append(b_target_q)
                    start_idx += b_dim
                target_q = torch.cat(target_q_list, dim=1).mean(dim=1, keepdim=True)
            else:
                next_actions = torch.argmax(next_q_eval, dim=1, keepdim=True)
                max_next_q = next_q_target.gather(1, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Loss calculation (Branching support)
        q_vals_all = self.q_eval(states, mfs)
        if self.branch_dims:
            # actions shape: [batch, num_branches]
            current_q_list = []
            start_idx = 0
            for i, b_dim in enumerate(self.branch_dims):
                b_q_vals = q_vals_all[:, start_idx : start_idx + b_dim]
                b_actions = actions[:, i : i + 1]
                current_q_list.append(b_q_vals.gather(1, b_actions))
                start_idx += b_dim
            current_q = torch.cat(current_q_list, dim=1).mean(dim=1, keepdim=True)
        else:
            current_q = q_vals_all.gather(1, actions)
            
        q_loss = nn.MSELoss()(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1.0)
        self.q_optimizer.step()

        # Soft update target network
        self.soft_update(self.q_eval, self.q_target, self.tau)
        
        return q_loss.item(), mf_loss.item(), current_q.mean().item()

    def save_model(self, path):
        torch.save({
            'q_eval': self.q_eval.state_dict(),
            'mf_net': self.mf_net.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.q_eval.load_state_dict(checkpoint['q_eval'])
        self.mf_net.load_state_dict(checkpoint['mf_net'])
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def decay_lr(self, factor):
        for param_group in self.q_optimizer.param_groups:
            param_group['lr'] *= factor
        for param_group in self.mf_optimizer.param_groups:
            param_group['lr'] *= factor
        return self.q_optimizer.param_groups[0]['lr']