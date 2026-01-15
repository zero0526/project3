import yaml
import os
from typing import Dict, Any, Optional
import torch

class Config:
    def __init__(self, network: str = 'atlanta', network_types: Optional[list] = None):
        """
        Initialize config with support for multiple network types.
        
        Args:
            network: Network topology name (e.g., 'atlanta', 'ta2')
            network_types: List of network types to load (e.g., ['cloud', 'edge', 'network', 'relay'])
                          If None, loads all available types
        """
        network_path = os.path.join("configs", "network_params.yaml")
        service_path = os.path.join("configs", "services.yaml")
        sim_path = os.path.join("configs", "simulation.yaml")
        
        # 1. Load network_params.yaml
        with open(network_path, 'r') as f:
            self.network = yaml.safe_load(f)
            
        # 2. Load simulation.yaml
        with open(sim_path, 'r') as f:
            sim_data = yaml.safe_load(f) or {}
        
        with open(service_path, 'r') as f:
            self.services= yaml.safe_load(f) or {}
        # Parse simulation config
        self.simulation = sim_data.get('simulation', {})
        self.sim_seed = self.simulation.get('seed', 42)
        self.sim_paths = self.simulation.get('paths', {})

        # 3. Filter nodes by network types
        all_nodes = self.network.get('nodes', {})
        
        if network_types is None:
            # Load all node types
            self.nodes = all_nodes
        else:
            # Load only specified network types
            self.nodes = {
                node_name: node_config 
                for node_name, node_config in all_nodes.items()
                if node_config.get('type') in network_types
            }
        
        # Get links and energy configuration
        self.links = self.network.get('links', {})
        self.energy = self.network.get('energy', {})
        self.lypa_coef = float(self.network.get('lypa_coef', 1e-7))
        self.cold_start_time = self.network.get('cold_start_time', {})
        self.avg_req = self.network.get('avg_req', 20)
        
        # Get coordinates for the selected network
        coordinates = self.network.get('coordinate', {})
        self.topology_name = network
        # Handle typo variants in config (altanta/atlanta)
        self.node_coordinates = (
            coordinates.get(network.lower(), {}) or 
            coordinates.get('altanta' if network.lower() == 'atlanta' else network.lower(), {})
        )
        self.neuron_net= self.simulation.get('NEURON_NET', {})
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_param= self.simulation["task"]
        
        # Helper paths for Converter
        self.TOPOLOGY_XML = self.sim_paths.get(f"topology_{self.topology_name.lower()}_xml")
        self.TOPOLOGY_JSON = self.sim_paths.get(f"topology_{self.topology_name.lower()}_json")
        
    def get_node_types(self) -> list:
        """Get list of loaded node types"""
        return list(set(node.get('type') for node in self.nodes.values()))
    
    def get_nodes_by_type(self, node_type: str) -> Dict[str, Any]:
        """Get all nodes of a specific type"""
        return {
            name: config 
            for name, config in self.nodes.items()
            if config.get('type') == node_type
        }
    
    def filter_by_network_types(self, network_types: list) -> 'Config':
        """
        Create a new Config instance filtered by network types.
        
        Args:
            network_types: List of network types to include
            
        Returns:
            New Config instance with filtered nodes
        """
        return Config(
            network=self.topology_name, 
            network_types=network_types
        )
    
    def get_service_details(self, service_id: int) -> Optional[Dict]:
        """Get details for a specific service by ID."""
        services_list = self.services.get('services', [])
        return next((s for s in services_list if s.get('id') == service_id), None)

        
cfg = Config()
