from src.utils import cfg
print(f"cfg.network type: {type(cfg.network)}")
print(f"cfg.network['lypa_coef'] type: {type(cfg.network.get('lypa_coef'))}")
print(f"cfg.network['lypa_coef'] value: {cfg.network.get('lypa_coef')}")
