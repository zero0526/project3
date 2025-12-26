import numpy as np
import networkx as nx 

class ChannelModel:
    """
    Mô hình kênh truyền vật lý (Physical Layer).
    Chịu trách nhiệm tính toán Delay, Băng thông thực tế và Tỉ lệ lỗi (nếu có).
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        # Các tham số mặc định nếu không có trong config
        self.default_prop_delay = self.config.get('propagation_delay', 0.001) # 1ms

    def compute_link_delay(self, data_size_mb, bandwidth_mbps, propagation_delay_s=None):
        """
        Tính trễ trên 1 liên kết đơn lẻ.
        T_link = T_trans + T_prop
        """
        if bandwidth_mbps <= 0:
            return float('inf')
            
        prop_delay = propagation_delay_s if propagation_delay_s is not None else self.default_prop_delay
        
        # 1. Trễ truyền dẫn (Transmission Delay): Thời gian đẩy gói tin ra đường truyền
        # Đổi MB -> Mb (Megabits)
        size_mbits = data_size_mb * 8
        t_trans = size_mbits / bandwidth_mbps
        
        # 2. Trễ lan truyền (Propagation Delay): Thời gian ánh sáng đi trong dây/không khí
        t_prop = prop_delay
        
        return t_trans + t_prop

    def compute_path_delay(self, graph: nx.Graph, path, data_size_mb):
        """
        Tính tổng trễ trên toàn bộ đường đi (End-to-End Latency).
        Áp dụng Công thức (10) trong bài báo: Sum(hops).
        
        Args:
            graph: NetworkX graph (để lấy thuộc tính từng cạnh).
            path: List các node ID [src, n1, n2, ..., dst].
            data_size_mb: Kích thước dữ liệu.
        """
        total_delay = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = graph.get_edge_data(u, v)
            
            bw = edge_data.get('bandwidth', 1000) # Mbps
            lat = edge_data.get('latency', self.default_prop_delay) # Seconds
            
            # Cộng dồn trễ
            link_delay = self.compute_link_delay(data_size_mb, bw, lat)
            
            if link_delay == float('inf'):
                return float('inf')
                
            total_delay += link_delay
            
        return total_delay

    def estimate_transmission_energy(self, total_delay, power_coeff=None):
        """
        Tính năng lượng truyền dẫn (Eq. 11).
        E = P * t (Đơn giản hóa)
        """
        p_coeff = power_coeff if power_coeff is not None else self.config.get('transmission_power', 0.2)
        return p_coeff * total_delay