import torch
import torch.optim as optim
import torch.nn as nn
from src.agents.networks import DuelingDQN, MeanFieldNet
from src.agents.base_agent import HMFD3QNBaseAgent, ReplayBuffer
from src.utils.config_loader import cfg

class LowerAgent(HMFD3QNBaseAgent):
    def __init__(self, state_dim, num_nodes, num_models_per_service, device="cpu", lr=1e-4):
        super().__init__(device)
        
        # Action Space: Chọn Node nào (V) VÀ Chọn Model nào (B)
        # Giả sử Flatten action space: |V| * |B|
        # Hoặc đơn giản hóa: Chọn Node thôi, Model chọn bằng logic (Greedy)
        # Theo bài báo: "Action ... including task offloading node AND model selection"
        self.action_dim = num_nodes * num_models_per_service
        self.local_state_dim = 3 + (2 * num_nodes)
        # Mean Field: Vector trung bình hành động của các terminal khác
        self.mf_dim = self.action_dim 

        # Networks
        self.q_eval = DuelingDQN(state_dim + self.mf_dim, self.action_dim).to(device)
        self.q_target = DuelingDQN(state_dim + self.mf_dim, self.action_dim).to(device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        
        self.mf_net = MeanFieldNet(state_dim + self.mf_dim, self.mf_dim).to(device)
        
        self.q_optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer(cfg.LOWER_BUFFER_CAPACITY)
        self.batch_size = cfg.LOWER_BATCH_SIZE
        self.gamma = cfg.LOWER_GAMMA
        self.temp = cfg.LOWER_TEMP # Temperature giảm dần trong quá trình train
        self.tau = cfg.LOWER_TAU

    def get_action(self, state, prev_mf):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prev_mf_t = torch.FloatTensor(prev_mf).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_mf = self.mf_net(state_t, prev_mf_t)
            q_values = self.q_eval(state_t, pred_mf)
            action_idx = self.select_action_boltzmann(q_values, self.temp)
            
        return action_idx, pred_mf.cpu().numpy()[0]

    def train(self):
        # (Logic train y hệt Upper Agent, chỉ khác Hyperparams)
        if len(self.buffer) < self.batch_size: return
        
        s, a, r, s_next, done, mf, mf_next = self.buffer.sample(self.batch_size)
        
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        mf = torch.FloatTensor(mf).to(self.device)
        mf_next = torch.FloatTensor(mf_next).to(self.device)

        # 1. Update MF Net
        self.mf_optimizer.zero_grad()
        pred_mf = self.mf_net(s, mf)
        mf_loss = nn.MSELoss()(pred_mf, mf_next)
        mf_loss.backward()
        self.mf_optimizer.step()

        # 2. Update Double Dueling DQN
        with torch.no_grad():
            next_actions = self.q_eval(s_next, mf_next).argmax(dim=1, keepdim=True)
            q_target_next = self.q_target(s_next, mf_next).gather(1, next_actions)
            target = r + (1 - done) * self.gamma * q_target_next

        q_current = self.q_eval(s, mf).gather(1, a)
        q_loss = nn.MSELoss()(q_current, target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        self.soft_update(self.q_eval, self.q_target, self.tau)
        
        # Anneal Temperature (Giảm nhiệt độ để bớt ngẫu nhiên)
        self.temp = max(0.1, self.temp * 0.9995)
        
        return q_loss.item(), mf_loss.item()