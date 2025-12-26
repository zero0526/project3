import yaml
import os

class Config:
    def __init__(self):
        # Paths to config files
        network_path = os.path.join("configs", "network_params.yaml")
        simulation_path = os.path.join("configs", "simulation.yaml")
        services_path = os.path.join("configs", "services.yaml")

        # 1. Load network_params.yaml
        with open(network_path, 'r') as f:
            self.network = yaml.safe_load(f)
            self.nodes = self.network.get('nodes', {})
            self.links = self.network.get('links', {})
            
            # Common link params
            self.DEFAULT_BANDWIDTH = self.links.get('default_bandwidth', 600.0)
            self.PROPAGATION_DELAY = self.links.get('propagation_delay', 0.005)
            self.TRANSMISSION_POWER = self.links.get('transmission_power', 0.2)

        # 2. Load simulation.yaml
        with open(simulation_path, 'r') as f:
            self.sim = yaml.safe_load(f).get('simulation', {})
            self.paths = self.sim.get('paths', {})
            
            # Paths
            self.TOPOLOGY_XML = self.paths.get('topology_xml', "configs/topologies/atlanta.xml")
            self.TOPOLOGY_JSON = self.paths.get('topology_json', "configs/topologies/atlanta_processed.json")
            self.LOGS_DIR = self.paths.get('logs', "data/logs/")

        # 3. Load services.yaml
        with open(services_path, 'r') as f:
            self.services = yaml.safe_load(f).get('services', [])

cfg = Config()
