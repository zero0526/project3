import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

def visualize_sndlib_with_coordinates(xml_path):
    # 1. Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Xử lý Namespace (SNDLib thường dùng http://sndlib.zib.de/network)
    ns = {'ns': 'http://sndlib.zib.de/network'}
    
    G = nx.Graph()
    pos = {} # Lưu tọa độ {node_id: (x, y)}

    # 2. Trích xuất Nodes và Tọa độ
    nodes_elem = root.find(".//ns:nodes", ns)
    for node in nodes_elem.findall("ns:node", ns):
        node_id = node.get('id')
        x = float(node.find("./ns:coordinates/ns:x", ns).text)
        y = float(node.find("./ns:coordinates/ns:y", ns).text)
        
        G.add_node(node_id)
        # Lưu tọa độ (x là kinh độ, y là vĩ độ)
        pos[node_id] = (x, y)

    # 3. Trích xuất Links
    links_elem = root.find(".//ns:links", ns)
    for link in links_elem.findall("ns:link", ns):
        source = link.find("ns:source", ns).text
        target = link.find("ns:target", ns).text
        G.add_edge(source, target)

    # 4. Vẽ đồ thị
    plt.figure(figsize=(12, 8))
    
    # Vẽ các cạnh
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray', alpha=0.5)
    
    # Vẽ các node
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', edgecolors='black')
    
    # Vẽ nhãn Node ID (kèm khoảng cách offset để không đè lên node)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title(f"SNDLib Topology: {xml_path} (Geographic Coordinates)", fontsize=14)
    plt.xlabel("Longitude (Kinh độ)")
    plt.ylabel("Latitude (Vĩ độ)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Thực thi
if __name__ == "__main__":
    # Đảm bảo file atlanta.xml nằm cùng thư mục
    visualize_sndlib_with_coordinates("./configs/topologies/atlanta.xml")
