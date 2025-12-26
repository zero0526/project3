# Cấu hình Hệ thống (Configurations)

Thư mục này chứa các file YAML định nghĩa tham số cho môi trường mô phỏng.

- **`simulation.yaml`**: Các tham số toàn cục như `seed`, số lượng `time_slots`, độ dài `time_frame`.
- **`network_params.yaml`**: Cấu hình băng thông mặc định, trễ truyền dẫn, và các hệ số kênh truyền.
- **`services.yaml`**: Định nghĩa 5 loại dịch vụ AI (Kích thước model, GFLOPS yêu cầu, Deadline, Loại Continuous/Occasional).