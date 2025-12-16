import networkx as nx
import xml.etree.ElementTree as ET
import json
import random
import os

class TopologyConverter:
    def __init__(self, source_file, output_file):
        self.source_file = source_file
        self.output_file = output_file
        self.graph = nx.Graph()

    def parse_sndlib_xml(self):
        """
        Đọc file XML chuẩn SNDlib và dựng đồ thị NetworkX.
        Hỗ trợ namespace và trích xuất tọa độ.
        """
        if not os.path.exists(self.source_file):
            print(f"Error: File {self.source_file} not found.")
            return

        tree = ET.parse(self.source_file)
        root = tree.getroot()

        # Handle Namespaces (SNDlib usually uses this namespace)
        # If your XML differs, adjust accordingly or use regex stripping (not recommended).
        ns_map = {}
        if 'http://sndlib.zib.de/network' in root.tag:
            ns_map = {'ns': 'http://sndlib.zib.de/network'}

        # Helper to find elements with or without namespace
        def find_all(element, path):
             if ns_map:
                 # Adjust path to include namespace prefix 'ns:' for each segment
                 # formatting like: .//ns:node is easier if path is simple
                 # But simplistic approach: Replace all words in path with ns:word
                 # Better: Use precise paths for SNDlib
                 return element.findall(path, ns_map)
             return element.findall(path)

        # 1. Parse Nodes
        # Path for SNDlib: network -> networkStructure -> nodes -> node
        # Using .//ns:node to find anywhere
        node_path = ".//ns:node" if ns_map else ".//node"
        
        nodes_found = root.findall(node_path, ns_map) if ns_map else root.findall(node_path)

        for node in nodes_found:
            node_id = node.get('id')
            
            # Extract coordinates
            x_val, y_val = 0.0, 0.0
            
            # Try to find coordinate elements
            # SNDlib: <coordinates> <x>...</x> <y>...</y> </coordinates>
            if ns_map:
                coord_elem = node.find("ns:coordinates", ns_map)
                if coord_elem is not None:
                    x_elem = coord_elem.find("ns:x", ns_map)
                    y_elem = coord_elem.find("ns:y", ns_map)
                    if x_elem is not None: x_val = float(x_elem.text)
                    if y_elem is not None: y_val = float(y_elem.text)
            else:
                coord_elem = node.find("coordinates")
                if coord_elem is not None:
                    x_elem = coord_elem.find("x")
                    y_elem = coord_elem.find("y")
                    if x_elem is not None: x_val = float(x_elem.text)
                    if y_elem is not None: y_val = float(y_elem.text)

            self.graph.add_node(node_id, pos=(x_val, y_val))

        # 2. Parse Links
        link_path = ".//ns:link" if ns_map else ".//link"
        links_found = root.findall(link_path, ns_map) if ns_map else root.findall(link_path)

        for link in links_found:
            if ns_map:
                source = link.find('ns:source', ns_map).text
                target = link.find('ns:target', ns_map).text
            else:
                source = link.find('source').text
                target = link.find('target').text
            
            # Mặc định chưa có băng thông, sẽ gán sau
            self.graph.add_edge(source, target)

        print(f"Parsed {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def assign_attributes(self, cloud_id=None, edge_ids=None):
        """
        Gán các thuộc tính mô phỏng (Băng thông, Loại node) theo bài báo.
        Optional: cloud_id (str), edge_ids (list of str) để cố định vị trí.
        """
        nodes = list(self.graph.nodes())
        
        if len(nodes) == 0:
            print("Graph is empty. Cannot assign attributes.")
            return

        # Tạo danh sách tạm để chọn node
        available_nodes = nodes.copy()
        
        # 1. Assign Cloud Node
        cloud_node = None
        if cloud_id and cloud_id in available_nodes:
            cloud_node = cloud_id
            available_nodes.remove(cloud_node)
        else:
            # Random pick
            if available_nodes:
                cloud_node = random.choice(available_nodes)
                available_nodes.remove(cloud_node)
        
        # 2. Assign Edge Nodes (Max 5 or remaining)
        assigned_edges = []
        if edge_ids:
            for eid in edge_ids:
                if eid in available_nodes:
                    assigned_edges.append(eid)
                    available_nodes.remove(eid)
        
        target_edges = 5
        needed = target_edges - len(assigned_edges)
        if needed > 0 and available_nodes:
            # Pick random needed count
            count = min(needed, len(available_nodes))
            picked = random.sample(available_nodes, count)
            assigned_edges.extend(picked)
            for p in picked:
                available_nodes.remove(p)

        # Remaining are relays
        relay_nodes = available_nodes

        # Gán thuộc tính Computing Capacity (GFLOPS) và Storage (GB)
        attrs = {}
        if cloud_node:
            attrs[cloud_node] = {"type": "cloud", "cpu": 5600, "ram": 80, "hdd": 500}
        
        for n in assigned_edges:
            attrs[n] = {"type": "edge", "cpu": 3000, "ram": 20, "hdd": 80}
            
        for n in relay_nodes:
            attrs[n] = {"type": "relay", "cpu": 0, "ram": 0, "hdd": 0} 

        nx.set_node_attributes(self.graph, attrs)

        # --- Cấu hình Link (Băng thông Mbps) ---
        # Bài báo: 50-100 MB/s => x8 => 400-800 Mbps
        MIN_BW = 400
        MAX_BW = 800
        
        link_attrs = {}
        for (u, v) in self.graph.edges():
            bandwidth = random.randint(MIN_BW, MAX_BW)
            link_attrs[(u, v)] = {"bandwidth": bandwidth, "latency_prop": 0.005}
            
        nx.set_edge_attributes(self.graph, link_attrs)

    def save_to_json(self):
        """Lưu ra file JSON để load vào môi trường giả lập."""
        data = nx.node_link_data(self.graph, edges="links")
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Topology saved to {self.output_file}")

# --- Test Script ---
if __name__ == "__main__":
    # Giả sử bạn đã tải file atlanta.xml từ SNDlib về folder configs/topologies
    # Nếu chưa có, tạo file dummy để test
    converter = TopologyConverter("configs/topologies/atlanta.xml", "configs/topologies/atlanta_sim.json")
    converter.parse_sndlib_xml() # Uncomment khi có file thật
    converter.assign_attributes(cloud_id='N1', edge_ids=['N2', 'N3']) # Ví dụ hardcode
    converter.save_to_json()
    print("Converter setup complete. Waiting for XML file.")