# AI Agents (Reinforcement Learning)

Cài đặt các thuật toán HMFD3QN.

- **`upper_agent.py`**: Agent quản lý các Computing Node. Quyết định `Service Placement` đầu mỗi Time Frame.
- **`lower_agent.py`**: Agent quản lý các Terminal. Quyết định `Task Offloading` và `Model Selection` mỗi Time Slot.
- **`mean_field.py`**: Mạng nơ-ron xấp xỉ trường trung bình (Mean Field Approximation Network) để ước lượng hành động của hàng xóm.