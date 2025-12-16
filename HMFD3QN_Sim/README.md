# HMFD3QN_Sim

This project simulates a hierarchical multi-agent reinforcement learning environment for dynamic service placement and offloading in a networked system.

## Directory Structure

- `configs/`: Contains configuration files for the simulation and services.
  - `topologies/`: Stores topology files in GraphML and JSON formats.
  - `simulation_config.yaml`: Parameters for the simulation (e.g., time slot, bandwidth, CPU, energy cost).
  - `services_config.yaml`: Configuration for AI services (e.g., size, GFLOPs).
- `data/`: Stores logs and models for training.
  - `logs/`: Raw logs in CSV format.
  - `models/`: Checkpoints for reinforcement learning agents.
- `src/`: Source code for the simulation.
  - `core/`: Core modules for the environment, network simulation, and workload generation.
  - `entities/`: Classes for nodes, terminals, and tasks.
  - `agents/`: Reinforcement learning logic for upper and lower agents.
  - `utils/`: Utility functions for mathematical operations and logging.
- `main.py`: Entry point for running the simulation.
- `requirements.txt`: Python dependencies for the project.

## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```
