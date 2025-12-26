import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanFieldNet(nn.Module):
    """
    Mô hình xấp xỉ trường trung bình (Eq. 43).
    Architecture: FC -> 50 (ReLU) -> 50 (ReLU) -> Output (Softmax)
    """
    def __init__(self, input_dim, output_dim):
        super(MeanFieldNet, self).__init__()
        # Input: Local State + Previous Mean Action
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc_out = nn.Linear(50, output_dim)

    def forward(self, local_state, prev_mean_action):
        # Nối vector trạng thái và hành động trung bình cũ
        x = torch.cat([local_state, prev_mean_action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        
        # Softmax output layer (theo mô tả bài báo) để ra phân phối xác suất hành động
        return F.softmax(x, dim=1)

class DuelingDQN(nn.Module):
    """
    Kiến trúc Dueling Q-Network.
    Input: State + Mean Field Vector (Eq. 42 xấp xỉ)
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # Lớp đặc trưng chung
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 1. Value Stream: Ước lượng giá trị của trạng thái V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 2. Advantage Stream: Ước lượng lợi thế của hành động A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state, mean_field):
        # Input là sự kết hợp giữa State và Mean Field
        x = torch.cat([state, mean_field], dim=1)
        
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Công thức kết hợp Dueling: Q(s,a) = V(s) + (A(s,a) - Mean(A))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals