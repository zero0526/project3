import numpy as np

class KKTSolver:
    def __init__(self, f_max_node, learning_rate=None, max_iter=100):
        self.f_max_node = f_max_node
        self.max_iter = max_iter

    def solve(self, G, Z, f_min_vec, f_max_vec):
        """
        Giải bài toán phân bổ tài nguyên dùng Bisection Search trên Lambda.
        Đã tối ưu cho trường hợp hằng số Z cực nhỏ.
        """
        num_services = len(G)
        if num_services == 0:
            return np.array([])

        # Nếu tổng f_min đã vượt quá năng lực node -> Scale f_min xuống
        sum_f_min = np.sum(f_min_vec)
        if sum_f_min >= self.f_max_node:
            return f_min_vec * (self.f_max_node / (sum_f_min + 1e-9))

        # --- Bisection Search cho Lambda (Giá tài nguyên) ---
        low = 0.0
        high = np.max(G) + 1.0
        Z_safe = np.maximum(Z, 1e-15)
        
        # Biến lưu trữ kết quả tốt nhất tìm được
        best_f = np.copy(f_min_vec)

        for _ in range(self.max_iter):
            mid = (low + high) / 2.0
            
            # f_i = (G_i - lambda) / 2Z_i
            f_cand = (G - mid) / (2 * Z_safe)
            f_cand = np.clip(f_cand, f_min_vec, f_max_vec)
            
            total_f = np.sum(f_cand)
            
            if total_f > self.f_max_node:
                # Nếu tổng f đang lớn hơn khả năng của node, ta phải tăng giá lambda lên
                low = mid
                # Lưu lại f_cand ở mức giá thấp này để scale sau cùng
                best_f = f_cand 
            else:
                # Nếu tổng f nhỏ hơn, giá đang quá cao, giảm giá lambda xuống
                high = mid
            
            # Nếu dải tìm kiếm đã quá nhỏ
            if abs(high - low) < 1e-12:
                break
        
        # --- NORMALIZATION ---
        # Sau khi tìm kiếm, best_f thường chứa tổng f hơi lớn hơn f_max_node 1 chút
        # Ta thực hiện scale xuống để đúng bằng 100% tài nguyên vật lý
        total_f = np.sum(best_f)
        if total_f > self.f_max_node:
            # Tỉ lệ scale
            ratio = self.f_max_node / total_f
            best_f = best_f * ratio
            # Sau khi scale, vẫn phải đảm bảo tối thiểu là f_min_vec
            best_f = np.maximum(best_f, f_min_vec)
            
        return best_f