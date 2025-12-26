import networkx as nx
import json
import yaml
import os
from channel_model import ChannelModel

class TopologyManager:
    def __init__(self, json_path, network_params_path):
        self.json_path = json_path
        self._load_network_params(network_params_path)
        self.graph = self._build_graph()
        self.channel = ChannelModel(self.network_config['links'])

    def _load_network_params(self, path):
        with open(path, 'r') as f:
            self.network_config = yaml.safe_load(f)
            self.default_bw = self.network_config['links']['default_bandwidth']
            self.default_lat = self.network_config['links']['propagation_delay']     # seconds

    def _build_graph(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Topology JSON not found at {self.json_path}. Run converter first.")

        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        G = nx.Graph()
        
        # Add Nodes with attributes
        for node in data['nodes']:
            G.add_node(node['id'], **node)
            
        # Add Edges
        for link in data['links']:
            G.add_edge(link['source'], link['target'], 
                       bandwidth=self.default_bw,
                       latency=self.default_lat)
        return G

    def get_nodes_by_type(self, node_type):
        """Lấy danh sách node theo loại (cloud/edge)"""
        return [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == node_type]

    def get_all_node_ids(self):
        return list(self.graph.nodes())
    
    def get_path_metrics(self, src, dst, data_size_mb):
        """
        Tìm đường và tính toán trễ thông qua ChannelModel.
        Output: (total_delay, hops, path)
        """
        try:
            # 1. Tìm đường ngắn nhất (Routing Logic)
            path = nx.shortest_path(self.graph, src, dst)
            hops = len(path) - 1
            
            if hops == 0:
                return 0.0, 0, path

            # 2. Tính toán Vật lý (Physics Logic) -> Gọi ChannelModel
            total_delay = self.channel.compute_path_delay(self.graph, path, data_size_mb)
            
            return total_delay, hops, path
            
        except nx.NetworkXNoPath:
            return float('inf'), float('inf'), []
        
    def get_logical_neighbors(self, computing_node_ids, max_hops=3):
        """
        Tìm các Computing Node lân cận (Logical Neighbors) cho thuật toán Mean Field.
        Bỏ qua các Relay Node trung gian.
        
        Args:
            computing_node_ids (list): Danh sách ID của các node có Agent (Edge/Network/Cloud).
            max_hops (int): Bán kính tìm kiếm (số bước nhảy tối đa).
            
        Returns:
            dict: {node_id: [neighbor_id_1, neighbor_id_2, ...]}
        """
        logical_neighbors = {}
        
        # Chuyển list thành set để tra cứu O(1)
        computing_set = set(computing_node_ids)
        
        for src in computing_node_ids:
            # 1. Tìm tất cả các node trong bán kính max_hops
            # Trả về dict {node_id: distance}
            nearby_nodes_dist = nx.single_source_shortest_path_length(
                self.graph, source=src, cutoff=max_hops
            )
            
            # 2. Lọc kết quả
            valid_neighbors = []
            for nid in nearby_nodes_dist:
                # Bỏ qua chính mình
                if nid == src:
                    continue
                    
                # Chỉ giữ lại node nếu nó là Computing Node (có Agent)
                if nid in computing_set:
                    valid_neighbors.append(nid)
            
            logical_neighbors[src] = valid_neighbors
            
        return logical_neighbors