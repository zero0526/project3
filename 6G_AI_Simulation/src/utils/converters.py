import networkx as nx
import xml.etree.ElementTree as ET
import json
import os
from typing import Dict, Any, List

class SNDLibConverter:
    def __init__(self, xml_path: str, json_path: str, node_config: Dict[str, Any]):
        self.xml_path = xml_path
        self.json_path = json_path
        self.node_config = node_config

    def convert(self) -> None:
        """convert old network to new net has cloud node, replay note, edge node"""
        if os.path.exists(self.xml_path):
            print(f"Found XML file: {self.xml_path}. Parsing...")
            self._parse_xml()
        else:
            print(f"XML file not found at {self.xml_path}.")
            raise FileNotFoundError(f"Cannot find topology file: {self.xml_path}")

    def _parse_xml(self) -> None:
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            ns: Dict[str, str] = {}
            if '}' in root.tag:
                ns_url = root.tag.split('}')[0].strip('{')
                ns = {'ns': ns_url}
            
            all_node_ids = []
            search_path = ".//ns:node" if ns else ".//node"
            xml_nodes = root.findall(search_path, ns)
            
            for node_elem in xml_nodes:
                nid = node_elem.get('id')
                if nid:
                    all_node_ids.append(nid)

            # --- step 2: distribute NODE---
            #require: 1 Cloud, 2 Network, 5 Edge and the rest is Relay.
            node_types_map = self._distribute_node_types(all_node_ids)

            # --- step 3: CREATE LIST NODES WITH SPECS ---
            final_nodes: List[Dict[str, Any]] = []
            
            for nid in all_node_ids:
                n_type = node_types_map.get(nid, 'relay')
                
                node_data = {
                    "id": nid,
                    "type": n_type
                }
                
                # HAS specs (CPU, RAM...) if it is Computing Node
                if n_type in ['cloud', 'network', 'edge']:
                    if n_type in self.node_config:
                        node_data.update(self.node_config[n_type])
                    else:
                        print(f"⚠️ Warning: Missing config for type '{n_type}'. Using default edge config.")
                        node_data.update(self.node_config.get('edge', {}))
                
                final_nodes.append(node_data)

            # --- step 4: PARSE LINKS ---
            links: List[Dict[str, str]] = []
            link_path = ".//ns:link" if ns else ".//link"
            src_path = "ns:source" if ns else "source"
            tgt_path = "ns:target" if ns else "target"

            for link_elem in root.findall(link_path, ns):
                src_elem = link_elem.find(src_path, ns)
                tgt_elem = link_elem.find(tgt_path, ns)
                
                if src_elem is not None and tgt_elem is not None:
                    # extra bandwidth/latency from XML, 
                    # currently TopologyManager load default from YAML
                    links.append({
                        "source": src_elem.text,
                        "target": tgt_elem.text
                    })

            self._save_json({"nodes": final_nodes, "links": links})
            
        except ET.ParseError as e:
            print(f"❌ XML Parse Error: {e}")
        except Exception as e:
            print(f"❌ General Error during parsing: {e}")

    def _distribute_node_types(self, node_ids: List[str]) -> Dict[str, str]:
        """
        distribute 8 Computing Nodes into Topology.
        """
        total_nodes = len(node_ids)
        type_map = {}
        
        assigned_indices = set()

        # 1. CLOUD (1 center node)
        cloud_idx = 0
        type_map[node_ids[cloud_idx]] = 'cloud'
        assigned_indices.add(cloud_idx)

        # 2. NETWORK NODES (2 Nodes) at 1/3 and 2/3 list nodes
        net_step = total_nodes // 3
        for i in range(1, 3):
            idx = (i * net_step) % total_nodes
            while idx in assigned_indices: 
                idx = (idx + 1) % total_nodes
            
            type_map[node_ids[idx]] = 'network'
            assigned_indices.add(idx)
        
        remaining_indices = [i for i in range(total_nodes) if i not in assigned_indices]
        
        # Nếu tổng node < 8 (trường hợp test), gán hết thành edge
        num_edges_needed = 5
        
        if len(remaining_indices) < num_edges_needed:
            # Fallback id
            for idx in remaining_indices:
                type_map[node_ids[idx]] = 'edge'
        else:
            # 5 edge node
            edge_step = len(remaining_indices) // num_edges_needed
            for k in range(num_edges_needed):
                rem_idx = k * edge_step
                real_idx = remaining_indices[rem_idx]
                
                type_map[node_ids[real_idx]] = 'edge'
        
        # In ra thống kê để kiểm tra
        print(f"📊 Node Distribution for {len(node_ids)} nodes:")
        counts = {'cloud': 0, 'network': 0, 'edge': 0, 'relay': 0}
        for nid in node_ids:
            t = type_map.get(nid, 'relay')
            counts[t] += 1
        print(f"   Cloud: {counts['cloud']}, Network: {counts['network']}, "
              f"Edge: {counts['edge']}, Relay: {counts['relay']}")

        return type_map

    def _save_json(self, data: Dict[str, Any]) -> None:
        """Lưu dữ liệu ra file JSON."""
        try:
            directory = os.path.dirname(self.json_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
                
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Topology saved to: {self.json_path}")
        except IOError as e:
            print(f"❌ File Write Error: {e}")

if __name__ == "__main__":
    dummy_node_config = {
        'cloud': {'cpu': 5000, 'ram': 100},
        'network': {'cpu': 3000, 'ram': 50},
        'edge': {'cpu': 1000, 'ram': 20}
    }
    
    converter = SNDLibConverter("atlanta.xml", "atlanta.json", dummy_node_config)
    converter.convert()