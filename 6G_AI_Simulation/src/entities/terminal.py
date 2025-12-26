# File: src/entities/terminal.py

class Terminal:
    def __init__(self, term_id, location_node_id, config=None):
        self.id = term_id
        
        # Vị trí hiện tại trong Topology (đang gắn vào Edge Node nào)
        self.location = location_node_id 
        
        # Các thuộc tính mở rộng cho tương lai
        self.battery_level = 100.0  # %
        self.is_mobile = False      # Có di chuyển không?
        
    def move(self, new_location_id):
        """Mô phỏng di chuyển sang vùng phủ sóng của Node khác"""
        self.location = new_location_id
        # print(f"Terminal {self.id} moved to {new_location_id}")

    def consume_battery(self, amount):
        self.battery_level = max(0, self.battery_level - amount)

    def __repr__(self):
        return f"<Terminal {self.id} @ {self.location}>"