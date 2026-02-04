import numpy as np

class KKTSolver:
    def __init__(self, f_max_node, learning_rate=None, max_iter=100):
        """
        f_max_node: f_v(tau) - Tổng tài nguyên tính toán tối đa của node v
        learning_rate: beta - Bước nhảy (step size) cho thuật toán subgradient
        max_iter: Số lần lặp tối đa để hội tụ
        """
        self.f_max_node = f_max_node
        self.max_iter = max_iter
        # Nếu không có learning_rate, đặt mặc định nhỏ để đảm bảo hội tụ
        self.learning_rate = learning_rate if learning_rate is not None else 0.01

    def solve(self, G, Z, f_min_vec, f_max_vec):
        """
        Giải bài toán tối ưu lồi P1-1 cho một node v.
        G, Z, f_min_vec, f_max_vec là các mảng (array) tương ứng với các dịch vụ s trên node đó.
        """
        num_services = len(G)
        
        # 1. Khởi tạo các nhân tử Lagrange (Lagrange multipliers)
        lam = 0.0
        mu_min = np.zeros(num_services)
        mu_max = np.zeros(num_services)

        f_optimized = np.zeros(num_services)
        
        # Đảm bảo Z không quá nhỏ để tránh chia cho 0 hoặc giá trị cực lớn
        Z_safe = np.maximum(Z, 1e-4)

        for _ in range(self.max_iter):
            # 2. Tính f_v,s hiện tại dựa trên điều kiện KKT (Eq 28)
            # f = (G - lambda - mu_max + mu_min) / (2 * Z)
            f_optimized = (G - lam - mu_max + mu_min) / (2 * Z_safe)
            
            # Clipping f_optimized để giữ cho gradient ổn định trong quá trình lặp
            # Không để f vượt quá 2 lần capacity để tránh divergence
            f_optimized = np.clip(f_optimized, 0, self.f_max_node * 2)

            # 3. Cập nhật các nhân tử Lagrange theo phương pháp Subgradient
            grad_lam = np.sum(f_optimized) - self.f_max_node
            lam = np.clip(lam + self.learning_rate * grad_lam, 0, 1e12)

            grad_mu_min = f_min_vec - f_optimized
            mu_min = np.clip(mu_min + self.learning_rate * grad_mu_min, 0, 1e12)

            grad_mu_max = f_optimized - f_max_vec
            mu_max = np.clip(mu_max + self.learning_rate * grad_mu_max, 0, 1e12)

        # Trả về phân bổ thực tế cuối cùng (đảm bảo f_min <= f <= f_max và tổng <= f_max_node)
        final_f = np.clip(f_optimized, f_min_vec, f_max_vec)
        sum_f = np.sum(final_f)
        if sum_f > self.f_max_node:
            # Tỷ lệ hóa nếu vượt quá tổng dung lượng node
            final_f = final_f * (self.f_max_node / sum_f)
            
        return final_f