import networkx as nx
from typing import List, Dict, Tuple
import json 
from src.utils import cfg

class TopologyManager:
    def __init__(self):
        self.graph = nx.Graph()
        self._path_cache = {}
        self.network_stats = {} 

    def load_topology_from_data(self):
        self.graph.clear()
        self._path_cache = {}
        topo_key = f"topology_{cfg.topology_name.lower()}_json"
        with open(cfg.sim_paths[topo_key], 'r') as f:
            data = json.load(f)
        self.global_config = data.get('global_config', {})
        self.network_stats = data.get('stats', {})

        # 2. Parse Nodes
        for node in data['nodes']:
            self.graph.add_node(
                node['id'],
                type=node.get('type', 'relay'),    # edge, cloud, network, relay
                
                cpu_available=node.get('cpu', 0.0),          
                ram_capacity=node.get('ram', 0.0),   
                hdd_capacity=node.get('hdd', 0.0),   
                
                pos=(node['coordinates']['x'], node['coordinates']['y']),
                energy_coef=float(node.get('energy_coef', 0.0))
            )

        # 3. Parse Links
        for link in data['links']:
            self.graph.add_edge(
                link['source'],
                link['target'],
                id=link['id'],
                
                transmission_rate=link.get('tranmission_rate', 0.0), 
                
                energy_coef=link.get('energy_coef', 0.2)
            )

        print(f"Loaded Topology: {data.get('network_name')} "
              f"({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} links)")

    def get_shortest_path(self, source: str, target: str) -> List[str]:
        if source == target:
            return [source]

        cache_key = (source, target)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        try:
            path = nx.shortest_path(self.graph, source=source, target=target, weight=None)
            self._path_cache[cache_key] = path
            return path
        except nx.NetworkXNoPath:
            return []

    def get_link_transmission_rate(self, u: str, v: str) -> float:
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['transmission_rate']
        return 0.0

    def get_node_resources(self, node_id: str) -> Dict:
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        return {}
    
    def get_nodes_by_type(self, node_type='edge') -> List[str]:
        """
        Lấy ra danh sách các node có type là 'edge'
        """
        return [node_id for node_id, data in self.graph.nodes(data=True) if data.get('type') == node_type]
