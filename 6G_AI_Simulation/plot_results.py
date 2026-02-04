import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(csv_path="data/training_history.csv", save_dir="data/plots"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Reward Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], label='Total Reward', color='blue')
    plt.title('Training Reward Convergence')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/reward_history.png")
    plt.close()

    # 2. QoS Violation Rate Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['violation_rate'], label='SLA Violation Rate (%)', color='red')
    plt.title('QoS Violation Rate History')
    plt.xlabel('Episode')
    plt.ylabel('Violation Rate (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/qos_history.png")
    plt.close()

    # 3. Energy Consumption Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['energy'], label='Total Energy (J)', color='green')
    plt.title('Energy Consumption History')
    plt.xlabel('Episode')
    plt.ylabel('Energy (J)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/energy_history.png")
    plt.close()

    # 4. Losses Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['q_loss'], label='Q-Loss', color='orange')
    plt.plot(df['episode'], df['mf_loss'], label='Mean Field Loss', color='purple')
    plt.yscale('log')
    plt.title('Agent Training Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss (Log Scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/losses_history.png")
    plt.close()

    # 5. Dual Axis (QoS & Energy) - KEY CONVERGENCE SHOWCASE
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('QoS Success Rate (%)', color=color)
    # QoS = 100 - violation_rate
    qos_rate = 100 - df['violation_rate']
    ax1.plot(df['episode'], qos_rate, color=color, linewidth=2, marker='o', label='QoS (Success %)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Total Energy (J)', color=color)
    ax2.plot(df['episode'], df['energy'], color=color, linewidth=2, linestyle='--', marker='s', label='Energy (J)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('ALGORITHM CONVERGENCE: QoS vs Energy Efficiency')
    fig.tight_layout()
    plt.savefig(f"{save_dir}/convergence_dual_axis.png")
    plt.close()

    print(f"Success: All plots saved to {save_dir}")
    print(f"Featured: {save_dir}/convergence_dual_axis.png shows combined convergence.")

if __name__ == "__main__":
    plot_training_results()
