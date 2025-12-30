import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done, mf, mf_next):
        self.buffer.append((s, a, r, s_next, done, mf, mf_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

class HMFD3QNBaseAgent:
    def __init__(self, device):
        self.device = device
        
    def select_action_boltzmann(self, q_values, temperature=1.0):
        """
        Chính sách Boltzmann (Eq. 45): pi(a|s) = exp(Q/zeta) / sum(exp(Q/zeta))
        """
        q_vals = q_values.detach().cpu().numpy().flatten()
        
        exp_q = np.exp((q_vals - np.max(q_vals)) / temperature)
        probs = exp_q / np.sum(exp_q)
        
        action = np.random.choice(len(q_vals), p=probs)
        return action

    def soft_update(self, local_model, target_model, tau=0.005):
        """Soft update tham số mạng Target (Eq. 49)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)