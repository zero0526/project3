from src.utils import cfg
from src.core.environment import SixGEnvironment
env = SixGEnvironment(service_config=cfg.services['services'])
node = list(env.nodes.values())[0]
print(f"Node energy_coeff: {node.energy_coeff} (type: {type(node.energy_coeff)})")

# Let's also check arrival_load
node.slot_arrival_workload[0] = 10.0
# Try the multiplication
v_param = 1e-7
print(f"Test multiplication: {v_param * node.energy_coeff * 10.0}")
