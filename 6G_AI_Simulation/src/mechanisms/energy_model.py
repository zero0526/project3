class EnergyModel:
    @staticmethod
    def calc_computation_energy(epsilon_c, f_allocated, workload_total, omega, epsilon_cold, t_cold):
        """
        Tính năng lượng tính toán (Eq. 14).
        
        Args:
            epsilon_c: Hệ số năng lượng (epsilon_v^c) - ví dụ 5e-10
            f_allocated: Năng lực tính toán cấp cho service (f_v,s) - đơn vị GFLOPS
            workload_total: Tổng workload (batches * GFLOPS/batch) - tương ứng (d/D)*F
            omega: Loại dịch vụ (1: continuous, 0: occasional)
            epsilon_cold: Hệ số năng lượng cold start (epsilon_v_cold)
            t_cold: Thời gian cold start (t_s,v_cold)
        """
        # 1. Năng lượng động (Dynamic Energy)
        # E_dynamic = epsilon_v^c * Workload * f^2
        e_dynamic = epsilon_c * workload_total * (f_allocated ** 2)
        
        # 2. Năng lượng khởi động lạnh (Cold Start Energy)
        # Chỉ áp dụng khi omega = 0 (Occasional Service)
        e_cold = (1 - omega) * epsilon_cold * t_cold
        
        return e_dynamic + e_cold