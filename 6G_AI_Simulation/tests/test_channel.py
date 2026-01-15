import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

try:
    from src.utils import cfg
    from src.network.channel_model import ChannelModel
    
    print("Checking Config...")
    topo_key = f"topology_{cfg.topology_name.lower()}_json"
    print(f"  Topology Name: {cfg.topology_name}")
    print(f"  Topology JSON Path: {cfg.sim_paths.get(topo_key)}")
    
    print("\nInitializing ChannelModel...")
    channel = ChannelModel()
    
    # Test path calculation from N1 to N15
    src, dst = "N5", "N7"
    data_mb = 10.0
    delay = channel.compute_path_delay(src, dst, data_mb)
    
    print(f"\nTest Path Calculation ({src} -> {dst}):")
    print(f"  Data Size: {data_mb} MB")
    print(f"  Shortest Path: {channel.topo.get_shortest_path(src, dst)}")
    print(f"  Total Delay: {delay:.6f} s")
    
    energy = channel.estimate_transmission_energy(delay)
    print(f"  Estimated Energy: {energy:.6f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
