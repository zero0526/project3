import torch
import yaml
import os
from src import hp
from src.core.environment import HMFD3QNEnv
from src.agents.upper_agent import UpperAgent
from src.agents.lower_agent import LowerAgent
from src.core.trainer import HMFD3QNTrainer # <--- Import class mới
from src.utils.converters import SNDLibConverter
from src.utils.math_utils import set_seed
from src.utils.config_loader import cfg

def setup_environment():
    """Setup Environment như cũ"""
    if not os.path.exists(cfg.TOPOLOGY_JSON):
        print("⚠️ Topology JSON not found. Running converter...")
        with open("configs/network_params.yaml") as f: net_conf = yaml.safe_load(f)
        converter = SNDLibConverter(
            cfg.TOPOLOGY_XML,
            cfg.TOPOLOGY_JSON,
            net_conf['nodes']
        )
        converter.convert()
    return HMFD3QNEnv("configs/simulation.yaml")

def main():
    # 1. Setup Global
    set_seed(hp.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== HMFD3QN SIMULATION STARTING ON {device.upper()} ===")

    # 2. Init Environment
    env = setup_environment()
    
    # 3. Init Agents
    num_services = len(env.service_config['services'])
    num_nodes = len(env.nodes)
    
    # Upper Agent (Placement)
    upper_agent = UpperAgent(
        state_dim=num_services, # State là backlog vector
        num_services=num_services, 
        device=device, 
        lr=hp.UPPER_LR
    )
    
    # Lower Agent (Offloading)
    lower_agent = LowerAgent(
        state_dim=4, # [Size, 0, Deadline, Acc]
        num_nodes=num_nodes, 
        num_models_per_service=1, 
        device=device, 
        lr=hp.LOWER_LR
    )

    # 4. Init Trainer
    trainer = HMFD3QNTrainer(env, upper_agent, lower_agent, device)

    # 5. Run Training
    trainer.train(total_episodes=hp.TOTAL_EPISODES)

if __name__ == "__main__":
    main()