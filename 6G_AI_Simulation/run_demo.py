import subprocess
import os
import time
import pandas as pd
from src.utils.logger import setup_logger

def run_demo():
    logger = setup_logger("Demo", "data/logs/demo.log")
    logger.info("Starting Short-Term Convergence Demo...")
    
    # 1. Cấu hình chạy nhanh (10 episodes)
    # Ta sẽ ghi đè tạm thời num_episodes trong test_agents nế cần, 
    # nhưng đơn giản nhất là báo user chạy test_agents.py rồi chạy cái này để tổng hợp.
    # Hoặc ta chạy trực tiếp test_agents.py ở đây.
    
    start_time = time.time()
    
    logger.info("Step 1: Running Agent Training (10 Episodes for quick demo)...")
    # Chúng ta sẽ thực thi test_agents.py nhưng giới hạn episodes
    # Để đơn giản, tôi sẽ yêu cầu user chạy test_agents.py (hiện tại là 30 ep, khá nhanh)
    # Nếu muốn cực nhanh, ta có thể sửa test_agents.py. 
    # Ở đây tôi chạy nguyên bản test_agents.py
    
    try:
        # Chạy test_agents.py như một process con
        subprocess.run(["python", "test_agents.py"], check=True)
    except Exception as e:
        logger.error(f"Error running test_agents.py: {e}")
        return

    logger.info("Step 2: Generating Convergence Plots...")
    try:
        subprocess.run(["python", "plot_results.py"], check=True)
    except Exception as e:
        logger.error(f"Error running plot_results.py: {e}")
        return

    # 3. Hiển thị kết quả tóm tắt
    if os.path.exists("data/training_history.csv"):
        df = pd.read_csv("data/training_history.csv")
        logger.info("\n" + "="*60)
        logger.info("DEMO RESULTS SUMMARY")
        logger.info("="*60)
        
        first_ep = df.iloc[0]
        last_ep = df.iloc[-1]
        best_ep = df.loc[df['violation_rate'].idxmin()]
        
        logger.info(f"Episodes Run: {len(df)}")
        logger.info(f"Initial QoS: {100 - first_ep['violation_rate']:.2f}% | Initial Energy: {first_ep['energy']:.2f}J")
        logger.info(f"Final QoS:   {100 - last_ep['violation_rate']:.2f}% | Final Energy:   {last_ep['energy']:.2f}J")
        logger.info(f"Best QoS:    {100 - best_ep['violation_rate']:.2f}% (Episode {int(best_ep['episode'])})")
        
        improvement_qos = (100 - last_ep['violation_rate']) - (100 - first_ep['violation_rate'])
        improvement_energy = first_ep['energy'] - last_ep['energy']
        
        logger.info("-" * 60)
        logger.info(f"QoS Improvement: {improvement_qos:+.2f}%")
        logger.info(f"Energy Reduction: {improvement_energy:+.2f}J ({(improvement_energy/first_ep['energy']*100) if first_ep['energy'] > 0 else 0:.1f}%)")
        logger.info("="*60)
        logger.info("Plots generated in: data/plots/")
        logger.info("Crucial Convergence Plot: data/plots/convergence_dual_axis.png")
    
    logger.info(f"Demo completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_demo()
