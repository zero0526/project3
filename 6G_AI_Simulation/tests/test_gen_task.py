from src.entities import Terminal
from src.network import TopologyManager
from src.core import WorkloadGenerator
from typing import List
from src.utils import cfg

topo= TopologyManager()
topo.load_topology_from_data()
task_cfg= cfg.simulation.get("task")
nodeIds: List[str]= topo.get_nodes_by_type(node_type='edge')
print(nodeIds)
terminals: List[Terminal]=[Terminal(f'terminal_{i}', nId, task_cfg.get("arrival_rate"), task_cfg.get("default_batch_size")) for i, nId in enumerate(nodeIds)] 
workGen= WorkloadGenerator(task_cfg, cfg.services.get("services"), terminals)
tasks= workGen.step(1)
print(len(tasks))
for t in tasks:
    print(t)
