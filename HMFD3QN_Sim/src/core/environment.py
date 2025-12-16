import gymnasium as gym

class HMFD3QNEnv(gym.Env):
    def __init__(self, config):
        self.network = NetworkSimulator(config['topology'])
        self.nodes = [Node(id, cap) for id in config['nodes']]
        self.terminals = [Terminal(id) for id in config['terminals']]
        self.current_slot = 0
        self.T = config['time_frame_duration'] # Ví dụ: 100 slots

    def step(self, actions_lower, action_upper=None):
        # 1. Thực hiện hành động Tầng trên (nếu là đầu Time Frame)
        if self.current_slot % self.T == 0 and action_upper:
            self._update_service_placement(action_upper)

        # 2. Sinh tác vụ mới (Workload Generation)
        tasks = self.workload_gen.generate(self.current_slot)

        # 3. Thực hiện hành động Tầng dưới (Task Scheduling)
        rewards, qos_metrics = [], []
        for terminal, action in zip(self.terminals, actions_lower):
            # Giao tiếp qua "Switch" ảo để tính trễ
            target_node = self.nodes[action['node_id']]
            trans_delay, _ = self.network.get_route_and_delay(terminal.id, target_node.id, task.size)
            
            # Đẩy vào hàng đợi Node và tính toán
            compute_delay, energy = target_node.process_task(task, action['model'])
            
            # Tổng hợp trễ
            total_delay = trans_delay + target_node.get_queueing_delay() + compute_delay
            
            # Ghi log QoS
            qos_metrics.append(self._check_sla(total_delay, task.deadline))

        # 4. Tính toán Mean Field (Giao tiếp ngang hàng)
        self._update_mean_fields()

        # 5. Phản hồi về tầng trên (Giao tiếp dọc)
        if (self.current_slot + 1) % self.T == 0:
            upper_reward = self._calculate_long_term_reward()
        
        self.current_slot += 1
        return next_state, rewards, done, info