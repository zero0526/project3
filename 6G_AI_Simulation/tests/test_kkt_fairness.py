import unittest
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from src.mechanisms.kkt_solver import KKTSolver

class TestKKTFairness(unittest.TestCase):
    def test_priority_allocation(self):
        f_max_node = 1000.0
        solver = KKTSolver(f_max_node)
        
        # Scenario: 3 Services
        # Svc 0: Hàng đợi nhỏ (1,000)
        # Svc 1: Hàng đợi trung bình (5,000)
        # Svc 2: Hàng đợi cực lớn (50,000)
        G = np.array([1000.0, 5000.0, 50000.0])
        Z = np.array([1e-10, 1e-10, 1e-10])
        f_min = np.array([0.0, 0.0, 0.0])
        f_max = np.array([1000.0, 1000.0, 1000.0])
        
        f_alloc = solver.solve(G, Z, f_min, f_max)
        
        print("\n[PHÂN TÍCH ƯU TIÊN]")
        print(f"Hàng đợi: {G}")
        print(f"Phân bổ CPU: {f_alloc}")
        print(f"Tổng CPU: {np.sum(f_alloc)}")
        
        # Kiểm tra: Svc 2 phải có CPU lớn nhất
        self.assertGreater(f_alloc[2], f_alloc[1])
        self.assertGreater(f_alloc[1], f_alloc[0])
        self.assertLessEqual(np.sum(f_alloc), f_max_node + 1e-5)

if __name__ == "__main__":
    unittest.main()
