# HMFD3QN_Sim

This project simulates a hierarchical multi-agent reinforcement learning environment for dynamic service placement and offloading in a networked system.

## Directory Structure

- `configs/`: Contains configuration files for the simulation and services.
  - `topologies/`: Stores topology files in GraphML and JSON formats.
  - `simulation_config.yaml`: Parameters for the simulation (e.g., time slot, bandwidth, CPU, energy cost).
  - `services_config.yaml`: Configuration for AI services (e.g., size, GFLOPs).
- `data/`: Stores logs and models for training.
  - `logs/`: Raw logs in CSV format.
  - `models/`: Checkpoints for reinforcement learning agents.
- `src/`: Source code for the simulation.
  - `core/`: Core modules for the environment, network simulation, and workload generation.
  - `entities/`: Classes for nodes, terminals, and tasks.
  - `agents/`: Reinforcement learning logic for upper and lower agents.
  - `utils/`: Utility functions for mathematical operations and logging.
- `main.py`: Entry point for running the simulation.
- `requirements.txt`: Python dependencies for the project.


------------------IMPLEMENTATION--------------------------------

📍 Giai đoạn 1: Cơ sở hạ tầng & Cấu hình (Infrastructure)
Mục tiêu: Thiết lập khung dự án và chuẩn hóa dữ liệu đầu vào.

  Setup: Khởi tạo cấu trúc thư mục và cài đặt thư viện (requirements.txt).

  Configs: Tạo các file YAML trong configs/:

    simulation.yaml: Tham số thời gian (slots, frames).

    services.yaml: Định nghĩa 5 loại dịch vụ AI (GFLOPS, Deadline).

  Topology Converter: Viết src/utils/converters.py để chuyển file XML SNDlib sang JSON và gán thuộc tính giả lập (CPU, RAM) cho các Node.

  Network Manager: Viết src/network/topology_manager.py sử dụng NetworkX để load đồ thị và tính đường đi ngắn nhất.

📍 Giai đoạn 2: Thực thể & Vật lý (Entities & Physics)
Mục tiêu: Các đối tượng trong mạng hoạt động đúng logic vật lý cơ bản.

  Task Object: Định nghĩa class Task trong src/entities/task.py.

  Workload Generator: Viết src/core/workload_generator.py sinh Task theo phân phối Zipf và Poisson.

  Computing Node: Viết khung class Node trong src/entities/node.py:

    Quản lý danh sách dịch vụ đã deploy.

    Quản lý hàng đợi (Queue) vào/ra.

    Logic nhận Task (Admit) và từ chối nếu chưa deploy dịch vụ.

📍 Giai đoạn 3: Cơ chế Toán học (Mathematical Core)
Mục tiêu: Hiện thực hóa các công thức tối ưu trong bài báo.

  Energy Model: Viết src/mechanisms/energy_model.py tính năng lượng truyền dẫn và tính toán (Eq. 11, 14).

  KKT Solver: Viết src/mechanisms/kkt_solver.py:

    Cài đặt công thức đóng (Closed-form) tính f*(Eq. 27).

    Cài đặt vòng lặp Subgradient cập nhật Lagrange Multipliers (Eq. 28-31).

  Integration: Tích hợp KKT Solver vào trong src/entities/node.py để Node tự tính lượng CPU cần cấp phát mỗi slot.
📍 Giai đoạn 4: Tích hợp Môi trường (Environment Integration)
Mục tiêu: Hoàn thiện môi trường Gym để chạy vòng lặp mô phỏng.

  Gym Env: Hoàn thiện src/core/environment.py:

    Hàm reset(): Khởi tạo lại mạng và node.

    Hàm step(): Điều phối luồng (Sinh Task -> Mạng -> Node -> Tính Reward).

    Xử lý logic 2 quy mô thời gian (Time Slot vs Time Frame).

  Simulation Loop: Viết main.py chạy thử với Random Action để kiểm tra dòng chảy dữ liệu (không crash).

📍 Giai đoạn 5: Hiển thị & Giám sát (Visualization)
Mục tiêu: Quan sát được hệ thống đang chạy thế nào (Digital Twin).

  Monitor: Viết src/utils/monitor.py chụp ảnh trạng thái (Snapshot) ra file JSON/CSV.

  Dashboard: Viết src/visualization/dashboard.py dùng Streamlit:

    Vẽ Topology mạng (đổi màu theo tải CPU).

    Vẽ Heatmap phân bố dịch vụ.

    Biểu đồ thời gian thực: Năng lượng & Vi phạm QoS.

📍 Giai đoạn 6: Trí tuệ nhân tạo (AI Agents)
Mục tiêu: Thay thế Random Action bằng thuật toán HMFD3QN.

  Mean Field Network: Viết src/agents/mean_field.py (Neural Network xấp xỉ hành động hàng xóm).

  Upper Agent: Viết src/agents/upper_agent.py (Service Placement).

  Lower Agent: Viết src/agents/lower_agent.py (Offloading).

  Training Loop: Cập nhật main.py để huấn luyện model.


  
## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   streamlit run src/visualization/dashboard.py
   ```
