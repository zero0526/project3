from src.utils import cfg
print(f"Num Services: {len(cfg.services['services'])}")
print(f"Num Nodes: {len(cfg.nodes)}")
for i, svc in enumerate(cfg.services['services']):
    print(f"Service {i}: {svc.get('name')} (id={svc.get('id')})")
