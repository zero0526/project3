import numpy as np

class MathOps:
    """
    Thư viện các công thức toán học từ bài báo HMFD3QN.
    """
    
    @staticmethod
    def calculate_transmission_energy(p_trans_coeff, data_size, bandwidth, hops):
        """
        Công thức (11): E_tr = P_i * time
        Time = (Data / Bandwidth) * Hops
        """
        if bandwidth <= 0: return float('inf')
        trans_time = (data_size / bandwidth) * hops
        return p_trans_coeff * trans_time

    @staticmethod
    def calculate_computation_energy(f_alloc, data_size, batch_size, model_complexity, coeff_sw, is_occasional, cold_cost):
        """
        Công thức (14): E_cp
        f_alloc: Tần số CPU phân bổ (GFLOPS)
        """
        # E_dynamic = coeff * (data/batch) * Complexity * f^2
        # Lưu ý: Bài báo công thức (14) có vẻ nhân f^2, nhưng về vật lý công suất ~ f^3 hoặc f^2. 
        # Ta tuân theo bài báo: E ~ time * Power ~ (Work/f) * (coeff * f^3) = Work * coeff * f^2
        
        workload = (data_size / batch_size) * model_complexity
        e_dynamic = coeff_sw * workload * (f_alloc ** 2)
        
        e_cold = cold_cost if is_occasional else 0
        return e_dynamic + e_cold

    @staticmethod
    def solve_resource_allocation_kkt(Q_vals, E_params, f_max, num_services, iterations=50, learning_rate=0.01):
        """
        Giải bài toán phân bổ tài nguyên bằng KKT (Mục IV-B, công thức 27-31).
        
        Input:
            Q_vals: Danh sách độ dài hàng đợi hiện tại [Q_{v,1}, Q_{v,2}...]
            E_params: Các tham số năng lượng/trọng số (G và Z trong Eq. 24)
            f_max: Tổng năng lực tính toán của Node
        
        Output:
            f_alloc: Mảng phân bổ tài nguyên tối ưu [f_{v,1}, f_{v,2}...]
        """
        # Khởi tạo Lagrange Multipliers
        lambda_v = 0.0
        mu_min = np.zeros(num_services)
        mu_max = np.zeros(num_services)
        f_alloc = np.zeros(num_services)
        
        # G(tau) và Z(tau) được tính từ Node trước khi gọi hàm này
        # Giả sử E_params chứa { 'G': [..], 'Z': [..] }
        G = np.array(E_params['G'])
        Z = np.array(E_params['Z'])
        
        # Subgradient Descent để tìm f* và multipliers tối ưu
        for _ in range(iterations):
            # 1. Cập nhật f (Eq. 27 & 28)
            # Tránh chia cho 0
            safe_Z = np.maximum(Z, 1e-6) 
            
            # Tính tử số
            numerator = G - lambda_v + mu_min - mu_max
            f_alloc = numerator / (2 * safe_Z)
            
            # Projection: f phải >= 0 và <= f_max (từng phần)
            # Tuy nhiên ràng buộc tổng f <= f_max nằm ở lambda
            f_alloc = np.maximum(f_alloc, 0)
            
            # 2. Cập nhật Lambda (Eq. 29) - Ràng buộc tổng tài nguyên
            total_f = np.sum(f_alloc)
            grad_lambda = total_f - f_max
            lambda_v = max(0, lambda_v + learning_rate * grad_lambda)
            
            # 3. Cập nhật Mu (Eq. 30, 31) - Ràng buộc biên
            # Ở đây ta đơn giản hóa: f_min = 0, f_max_s = f_max (hoặc một giới hạn logic nào đó)
            # Bài báo dùng f_alloc trực tiếp, ta chỉ cần đảm bảo f >= 0 (đã làm ở bước 1)
            # Nên bước cập nhật Mu có thể lược bỏ nếu dùng hàm max(0, ...) trực tiếp.
            
        # Chuẩn hóa cuối cùng để đảm bảo không vượt quá f_max cứng
        if np.sum(f_alloc) > f_max:
            f_alloc = f_alloc * (f_max / np.sum(f_alloc))
            
        return f_alloc