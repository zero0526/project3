import numpy as np

class KKTSolver:
    def __init__(self, f_max_node, learning_rate=0.01, max_iter=50):
        self.f_max_node = f_max_node  # Tổng dung lượng Node (f_v)
        self.lr = learning_rate       # Beta (Step size)
        self.max_iter = max_iter

    def solve(self, G, Z, f_min_vec, f_max_vec):
        """
        Giải bài toán tối ưu tài nguyên (Resource Allocation) theo KKT.
        
        Args:
            G (np.array): Vector áp lực hàng đợi G(tau).
            Z (np.array): Vector áp lực năng lượng Z(tau).
            f_min_vec (np.array): Vector f_min cho từng dịch vụ (Eq. 25 QoS).
            f_max_vec (np.array): Vector f_max cho từng dịch vụ (Thường = f_max_node).
            
        Returns:
            f_optimal (np.array): Phân bổ tài nguyên tối ưu.
        """
        num_services = len(G)
        if num_services == 0:
            return np.array([])

        # --- Khởi tạo Nhân tử Lagrange (Lagrange Multipliers) ---
        lambda_v = 0.0                    # Eq. 29 (Constraint tổng tài nguyên)
        mu_min = np.zeros(num_services)   # Eq. 30 (Constraint f >= f_min)
        mu_max = np.zeros(num_services)   # Eq. 31 (Constraint f <= f_max)
        
        f_alloc = np.zeros(num_services)

        # --- Vòng lặp Subgradient Descent ---
        for k in range(self.max_iter):
            # Beta giảm dần để hội tụ (tùy chọn, ở đây giữ cố định hoặc giảm nhẹ)
            beta = self.lr / (1 + 0.1 * k)

            # 1. Tính f(tau) tại bước k (Eq. 28)
            # f = (G - lambda + mu_min - mu_max) / 2Z
            numerator = G - lambda_v + mu_min - mu_max
            denominator = 2 * np.maximum(Z, 1e-9) # Tránh chia cho 0
            
            f_alloc = numerator / denominator
            
            # Lưu ý: f vật lý không thể âm, dù toán học có thể ra âm
            f_alloc = np.maximum(f_alloc, 0.0)

            # 2. Cập nhật Lambda (Eq. 29)
            # Gradient = Tổng f phân bổ - f_max_node
            # Nếu Tổng f > f_max_node -> Gradient dương -> Lambda tăng -> Giá đắt -> f giảm
            total_f = np.sum(f_alloc)
            grad_lambda = total_f - self.f_max_node
            lambda_v = max(0.0, lambda_v + beta * grad_lambda)

            # 3. Cập nhật Mu_min (Eq. 30)
            # Gradient = f_min - f_alloc
            # Nếu f_alloc < f_min (Vi phạm QoS) -> Gradient dương -> Mu_min tăng -> Kéo f lên
            grad_mu_min = f_min_vec - f_alloc
            mu_min = np.maximum(0.0, mu_min + beta * grad_mu_min)

            # 4. Cập nhật Mu_max (Eq. 31)
            # Gradient = f_alloc - f_max_vec
            # Nếu f_alloc > f_max_vec (Vi phạm giới hạn) -> Gradient dương -> Mu_max tăng -> Đè f xuống
            grad_mu_max = f_alloc - f_max_vec
            mu_max = np.maximum(0.0, mu_max + beta * grad_mu_max)

        return f_alloc