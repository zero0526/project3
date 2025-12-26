import yaml
from src.utils.converters import SNDLibConverter
from src.network.topology_manager import TopologyManager

def run_test():
    print("=== TESTING PHASE 1: INFRASTRUCTURE ===")
    
    # 1. Load Configs
    print("[1] Loading configurations...")
    with open("./configs/network_params.yaml") as f: net_conf = yaml.safe_load(f)
    with open("./configs/simulation.yaml") as f: sim_conf = yaml.safe_load(f)
    
    xml_path = sim_conf['simulation']['paths']['topology_xml']
    json_path = sim_conf['simulation']['paths']['topology_json']
    
    # 2. Run Converter
    print("[2] Running Topology Converter...")
    converter = SNDLibConverter(xml_path, json_path, net_conf['nodes'])
    converter.convert()
    
    # 3. Initialize Topology Manager
    print("[3] Initializing Topology Manager...")
    topo = TopologyManager(json_path, "./configs/network_params.yaml")
    
    # 4. Verify Data
    clouds = topo.get_nodes_by_type('cloud')
    edges = topo.get_nodes_by_type('edge')
    print(f"    -> Found {len(clouds)} Cloud nodes: {clouds}")
    print(f"    -> Found {len(edges)} Edge nodes: {edges}")
    
    if len(edges) > 0:
        src = edges[0]
        dst = clouds[0]
        # Test latency calculation for 10MB data
        delay, hops, path = topo.get_path_info(src, dst, data_size_mb=10)
        print(f"[4] Path Test ({src} -> {dst}):")
        print(f"    -> Path: {path}")
        print(f"    -> Hops: {hops}")
        print(f"    -> Estimated Latency (10MB): {delay:.4f}s")
        
    print("\n✅ PHASE 1 COMPLETE! Infrastructure is ready.")

if __name__ == "__main__":
    run_test()