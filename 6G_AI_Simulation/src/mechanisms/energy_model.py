
class EnergyModel:
    @staticmethod
    def calc_transmission(p_trans, data_size_mb, bandwidth_mbps, hops):
        """
        Tính năng lượng truyền dẫn (Eq. 11).
        E_tr = Power * Time * Hops
        """
        if bandwidth_mbps <= 0: return float('inf')
        
        # Đổi MB -> Mb
        data_size_mbits = data_size_mb * 8
        trans_time = data_size_mbits / bandwidth_mbps
        
        # E = P * t * hops
        return p_trans * trans_time * hops

    @staticmethod
    def calc_computation(coeff, frequency, duration, omega, cold_start_energy):
        """
        Tính năng lượng tính toán (Eq. 14).
        E_cp = E_dynamic + E_cold_start
        E_dynamic = coeff * f^2 * t (Theo bài báo để đảm bảo tính lồi)
        """
        # 1. Năng lượng động (Dynamic Energy)
        # Power = coeff * f^2 (hoặc f^3 tùy hardware, nhưng bài báo dùng mô hình convex)
        e_dynamic = coeff * (frequency ** 2) * duration
        
        # 2. Năng lượng khởi động lạnh (Cold Start)
        # Chỉ tính nếu là Occasional Service (omega=0)
        e_cold = (1 - omega) * cold_start_energy
        
        return e_dynamic + e_cold