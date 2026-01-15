import torch
import yaml
import os
import numpy as np
from src import hp
from src.core.environment import SixGEnvironment
from src.agents.train import HMFD3QN_Trainer
from src.utils.converters import SNDLibConverter
from src.utils.math_utils import set_seed
from src.utils import cfg

def setup_environment():
    """Khởi tạo môi trường mô phỏng."""
    # 1. Kiểm tra topology đã được convert chưa
    if not os.path.exists(cfg.TOPOLOGY_JSON):
        print("⚠️ Topology JSON not found. Running converter...")
        converter = SNDLibConverter(
            cfg.TOPOLOGY_XML,
            cfg.TOPOLOGY_JSON,
            cfg
        )
        converter.convert()
    
    # 2. Tạo môi trường
    env = SixGEnvironment(cfg.services['services'])
    return env

def main():
    # 1. Setup Global
    set_seed(cfg.simulation.get('seed', 42))
    device = cfg.device
    print(f"\n{'='*60}")
    print(f"{'HMFD3QN 6G SIMULATION TRAINING STARTING':^60}")
    print(f"{'Device: ' + str(device).upper():^60}")
    print(f"{'='*60}\n")

    # 2. Init Environment
    env = setup_environment()
    print(f"Environment Initialized:")
    print(f" - Total Nodes: {len(env.nodes)} (Agent Nodes: {len(env.agent_node_ids)}, Cloud: {len(env.cloud_node_ids)})")
    print(f" - Total Terminals: {len(env.terminals)}")
    print(f" - Services: {env.num_services}")
    
    # 3. Init Trainer
    # Trainer sẽ tự động khởi tạo Upper Agents cho agent_node_ids 
    # và Lower Agents cho terminals.
    trainer = HMFD3QN_Trainer(env)

    # 4. Run Training
    num_episodes = cfg.neuron_net.get('NUMOF_TRAIN_EP', 100)
    print(f"\nStarting training for {num_episodes} episodes...")
    try:
        trainer.train(num_episodes=num_episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # 5. Save Models
    trainer.save_models(path="data/checkpoints/")

if __name__ == "__main__":
    main()