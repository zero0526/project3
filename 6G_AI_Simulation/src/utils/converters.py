import xml.etree.ElementTree as ET
import json
import os
import random
from typing import Dict, Any, List, Optional

from .config_loader import cfg 

class SNDLibConverter:
    def __init__(self, 
                 xml_path: str, 
                 json_path: str, 
                 config: Any): # Nhận object Config
        """
        Khởi tạo bộ chuyển đổi topology.
        
        Args:
            xml_path: Đường dẫn file input .xml (SNDLib)
            json_path: Đường dẫn file output .json
            config: Instance của class Config chứa các tham số node/link/energy
        """
        self.xml_path = xml_path
        self.json_path = json_path
        self.cfg = config
        
        # Lấy mapping vị trí node dựa trên network name đã load trong Config
        # Ví dụ: {'cloud': ['N15'], 'edge': ['N4', 'N5']...}
        self.node_mapping = self.cfg.node_coordinates

    def convert(self) -> None:
        """Thực hiện chuyển đổi và lưu file JSON."""
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file topology: {self.xml_path}")
        
        print(f"🔄 Đang đọc file XML: {self.xml_path}...")
        print(f"   Network Mode: {self.cfg.topology_name.upper()}")
        
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # Xử lý Namespace (thường gặp trong SNDLib)
            ns = {}
            if '}' in root.tag:
                ns_url = root.tag.split('}')[0].strip('{')
                ns = {'ns': ns_url}

            # 1. Parse Nodes & Assign Specs
            nodes_data = self._parse_nodes(root, ns)
            
            # 2. Parse Links
            links_data = self._parse_links(root, ns)

            # 3. Tổng hợp dữ liệu đầu ra
            output_data = {
                "network_name": self.cfg.topology_name,
                "created_at": "Simulation Pre-processing",
                "stats": {
                    "total_nodes": len(nodes_data),
                    "total_links": len(links_data),
                    "node_types": self._count_types(nodes_data)
                },
                # Lưu global config vào json để tiện load sau này
                "global_config": {
                    "energy": self.cfg.energy,
                    "links_capacity": {
                        "min": self.cfg.links.get('tranmission_rate_min'),
                        "max": self.cfg.links.get('tranmission_rate_max')
                    },
                    "cold_start_time": self.cfg.cold_start_time
                },
                "nodes": nodes_data,
                "links": links_data
            }
            
            self._save_json(output_data)

        except ET.ParseError as e:
            print(f"❌ Lỗi cú pháp XML: {e}")
        except Exception as e:
            print(f"❌ Lỗi không xác định: {e}")
            raise e

    def _parse_nodes(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, Any]]:
        final_nodes = []
        # Tìm tất cả thẻ node
        xml_nodes = root.findall(".//ns:node", ns) if ns else root.findall(".//node")
        
        print(f"   Tìm thấy {len(xml_nodes)} nodes trong XML.")

        for node_elem in xml_nodes:
            nid = node_elem.get('id')
            
            # --- XỬ LÝ TỌA ĐỘ (COORDINATES) ---
            # Mặc định là (0,0) nếu không tìm thấy trong XML
            coordinates = {"x": 0.0, "y": 0.0}
            
            # Tìm thẻ <coordinates>
            coords_elem = node_elem.find("ns:coordinates", ns) if ns else node_elem.find("coordinates")
            
            if coords_elem is not None:
                try:
                    # Lấy giá trị x, y (có xử lý namespace con bên trong)
                    x_tag = coords_elem.find("ns:x", ns) if ns else coords_elem.find("x")
                    y_tag = coords_elem.find("ns:y", ns) if ns else coords_elem.find("y")
                    
                    if x_tag is not None and y_tag is not None:
                        coordinates = {
                            "x": float(x_tag.text),
                            "y": float(y_tag.text)
                        }
                except ValueError:
                    print(f"⚠️ Cảnh báo: Lỗi định dạng tọa độ tại node {nid}")

            # --- CÁC LOGIC KHÁC (Type, Specs) ---
            node_type = self._determine_node_type(nid)
            specs = self._generate_specs(node_type)

            # --- TẠO OBJECT NODE HOÀN CHỈNH ---
            node_obj = {
                "id": nid,
                "type": node_type,
                "coordinates": coordinates,  # <--- Đã thêm trường này
                **specs
            }
            final_nodes.append(node_obj)
            
        return final_nodes

    def _determine_node_type(self, nid: str) -> str:
        """
        Map node ID sang loại node dựa trên config `coordinate`.
        Mặc định là 'relay' nếu không tìm thấy.
        """
        # self.node_mapping dạng: {'cloud': ['N15'], 'edge': ['N4',...]}
        for type_name, id_list in self.node_mapping.items():
            if nid in id_list:
                return type_name
        return 'relay'

    def _generate_specs(self, node_type: str) -> Dict[str, Any]:
        """
        Lấy cấu hình cơ bản từ Config và sinh giá trị ngẫu nhiên cho CPU
        """
        # Lấy template config cho loại node này (vd: cloud, edge...)
        type_config = self.cfg.nodes.get(node_type)
        
        # Nếu không có config hoặc là relay -> trả về 0 hết
        if not type_config or node_type == 'relay':
            return {
                "cpu": 0.0,
                "ram": 0.0,
                "hdd": 0.0,
                "energy_coef": 0.0,
                "is_computing": False
            }
        
        # Sinh CPU ngẫu nhiên trong khoảng [min, max]
        cpu_min = type_config.get('cpu_min', 0)
        cpu_max = type_config.get('cpu_max', cpu_min)
        real_cpu = round(random.uniform(cpu_min, cpu_max), 2)
        
        # Lấy hệ số năng lượng chung
        e_coef = self.cfg.energy.get('computation_coef', 0)

        return {
            "cpu": real_cpu,                     # GFLOPS thực tế
            "ram": type_config.get('ram', 0),    # GB
            "hdd": type_config.get('hdd', 0),    # GB
            "energy_coef": e_coef,               # W/GFLOPS
            "is_computing": True
        }

    def _parse_links(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict[str, str]]:
        links = []
        xml_links = root.findall(".//ns:link", ns) if ns else root.findall(".//link")
        tranmission_rate_min= self.cfg.links.get('tranmission_rate_min')
        tranmission_rate_max= self.cfg.links.get('tranmission_rate_max')
        for link in xml_links:
            src = link.find("ns:source", ns).text if ns else link.find("source").text
            tgt = link.find("ns:target", ns).text if ns else link.find("target").text
            lid = link.get("id")
            
            links.append({
                "id": lid,
                "source": src,
                "target": tgt,
                "tranmission_rate": random.uniform(tranmission_rate_min, tranmission_rate_max),
                "energy_coef": self.cfg.energy.get('tranmission_coef')
            })
        return links

    def _count_types(self, nodes: List[Dict]) -> Dict[str, int]:
        counts = {}
        for n in nodes:
            t = n['type']
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _save_json(self, data: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Đã lưu topology tại: {self.json_path}")
        print(f"   Thống kê: {json.dumps(data['stats'], indent=2)}")

if __name__ == "__main__":
    converter = SNDLibConverter("./configs/topologies/ta2.xml", "./configs/topologies/ta2_processed.json", cfg)
    converter.convert()