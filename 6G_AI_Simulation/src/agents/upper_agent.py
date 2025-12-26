import torch
import torch.optim as optim
import torch.nn as nn
from src.agents.networks import DuelingDQN, MeanFieldNet
from src.agents.base_agent import HMFD3QNBaseAgent, ReplayBuffer
from src.utils.config_loader import cfg

class UpperAgent(HMFD3QNBaseAgent):
    def __init__(self, state_dim, num_services, device="cpu", lr=1e-4):
        super().__init__(device)
        self.num_services = num_services
        
        # Action Space: Binary vector cho m dịch vụ => 2^m actions
        self.action_dim = 2 ** num_services
        
        # Mean Field Input: Là vector xác suất triển khai trung bình (kích thước = action_dim)
        # Hoặc để đơn giản hóa theo bài báo: "observed local mean field" (Eq. 44)
        self.mf_dim = self.action_dim 

        # 1. Main Q-Networks (Double)
        self.q_eval = DuelingDQN(state_dim + self.mf_dim, self.action_dim).to(device)
        self.q_target = DuelingDQN(state_dim + self.mf_dim, self.action_dim).to(device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        
        # 2. Mean Field Approx Network (Eq. 43)
        self.mf_net = MeanFieldNet(state_dim + self.mf_dim, self.mf_dim).to(device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer(cfg.UPPER_BUFFER_CAPACITY)
        self.batch_size = cfg.UPPER_BATCH_SIZE
        self.gamma = cfg.UPPER_GAMMA
        self.temp = cfg.UPPER_TEMP # Boltzmann temperature (Zeta)
        self.tau = cfg.UPPER_TAU

    def get_action(self, state, prev_mean_field):
        """
        1. Dự đoán Mean Field hiện tại bằng MF_Net.
        2. Tính Q-values.
        3. Chọn action bằng Boltzmann.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prev_mf_t = torch.FloatTensor(prev_mean_field).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Bước 1: MF Net dự đoán
            pred_mf = self.mf_net(state_t, prev_mf_t)
            
            # Bước 2: Dueling DQN tính Q
            q_values = self.q_eval(state_t, pred_mf)
            
            # Bước 3: Boltzmann selection
            action_idx = self.select_action_boltzmann(q_values, self.temp)
            
        return action_idx, pred_mf.cpu().numpy()[0]

    def train(self):
        if len(self.buffer) < self.batch_size: return
        
        # Sample batch
        s, a, r, s_next, done, mf, mf_next = self.buffer.sample(self.batch_size)
        
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        mf = torch.FloatTensor(mf).to(self.device)
        mf_next = torch.FloatTensor(mf_next).to(self.device)

        # --- 1. Train Mean Field Net ---
        # Loss = MSE(Dự đoán MF hiện tại, MF thực tế quan sát được)
        # Lưu ý: Trong thực tế MF thực tế (target) được tính từ hàng xóm
        self.mf_optimizer.zero_grad()
        pred_mf = self.mf_net(s, mf) # mf ở đây đóng vai trò là prev_mf
        mf_loss = nn.MSELoss()(pred_mf, mf_next) # mf_next là observed MF thực tế
        mf_loss.backward()
        self.mf_optimizer.step()

        # --- 2. Train Dueling Double DQN ---
        # Tính Q_target theo Double DQN (Eq. 47)
        with torch.no_grad():
            # Dùng Q_eval để chọn hành động tốt nhất ở next state
            next_actions = self.q_eval(s_next, mf_next).argmax(dim=1, keepdim=True)
            # Dùng Q_target để tính giá trị của hành động đó
            q_target_next = self.q_target(s_next, mf_next).gather(1, next_actions)
            target = r + (1 - done) * self.gamma * q_target_next

        # Tính Q_eval hiện tại
        q_current = self.q_eval(s, mf).gather(1, a)
        
        q_loss = nn.MSELoss()(q_current, target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        self.soft_update(self.q_eval, self.q_target, self.tau)
        
        return q_loss.item(), mf_loss.item()