#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


plt.rcParams['axes.titlesize'] = 10 # subplot title
plt.rcParams['axes.labelsize'] = 8  # all subplot titles
plt.rcParams['figure.titlesize'] = 10 # suptitle
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.title_fontsize'] = 8

# Latency objective (seconds, $\bar{\rho}$ in TABLE I of MASCOTS 2025)
SLO_LATENCY = 1.5

# Maximum latency observed in the data (seconds)
MAX_LATENCY = 4.5

# CPU Utilization objective (percent, $\bar{u}$ in TABLE I of MASCOTS 2025)
SLO_CPU_UTILIZATION = 0.8

# By definition it is 1
MAX_CPU_UTILIZATION = 1.0

# Scaling factor ($\beta$ in TABLE I of MASCOTS 2025)
BETA = 1.0 / MAX_LATENCY

# Ellipse width and height
CW, CH = 0.12, 0.15

# Eq (2) of MASCOTS 2025
def performance_reward(latency):
    if latency > SLO_LATENCY:
        return 1 - np.exp(BETA * latency)
    else:
        return 0

def cost_reward(cpu_utilization):
    if cpu_utilization < SLO_CPU_UTILIZATION:
        return 1 - np.exp(1 - cpu_utilization)
    else:
        return 0


def total_reward(latency, cpu_utilization):
    return performance_reward(latency) + cost_reward(cpu_utilization)


def plot_reward_functions():
    pre_latencies = np.linspace(0, SLO_LATENCY, 5)

    # second piece does not cross x=SLO_LATENCY
    post_latencies = np.linspace(SLO_LATENCY + 0.0001, MAX_LATENCY, 50)

    pre_rewards = [performance_reward(l) for l in pre_latencies]
    post_rewards = [performance_reward(l) for l in post_latencies]

    fig = plt.figure(figsize=(7,3))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0,0])
    ax.axvline(SLO_LATENCY, linestyle='--', color='orange', label=r'SLO_LATENCY ($\bar{\rho}$)')

    ax.plot(pre_latencies, pre_rewards, color='tab:blue')
    ax.plot(post_latencies, post_rewards, color='tab:blue')
    ax.add_patch(patches.Ellipse((0, 0), CW, CH, fill=True, edgecolor='tab:blue'))
    ax.add_patch(patches.Ellipse((SLO_LATENCY, 0), CW, CH, zorder=10, fill=True, edgecolor='tab:blue'))
    ax.add_patch(patches.Ellipse((SLO_LATENCY, 1-np.exp(BETA*SLO_LATENCY)), CW, CH, zorder=10, fill=True, edgecolor='tab:blue', facecolor='white'))
    ax.add_patch(patches.Ellipse((MAX_LATENCY, 1-np.exp(BETA*MAX_LATENCY)), CW, CH, zorder=10, fill=True, edgecolor='tab:blue', facecolor='tab:blue'))


    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Reward')
    #ax.set_aspect("equal")
    ax.legend()
    ax.set_title('Performance reward')

    # Cost Reward
    ax = fig.add_subplot(gs[1,0])
    # second piece does not cross x=SLO_CPU_UTILIZAION
    pre_utilization = np.linspace(0, SLO_CPU_UTILIZATION - 0.00001, 50)
    post_utilization = np.linspace(SLO_CPU_UTILIZATION, MAX_CPU_UTILIZATION, 50)

    pre_rewards = [cost_reward(l) for l in pre_utilization]
    post_rewards = [cost_reward(l) for l in post_utilization]

    ax.axvline(SLO_CPU_UTILIZATION, linestyle='--', color='orange', label=r'SLO_CPU_UTILIZATION ($\bar{u}$)')

    ax.plot(pre_utilization, pre_rewards, color='tab:blue')
    ax.plot(post_utilization, post_rewards, color='tab:blue')
    ax.add_patch(patches.Ellipse((0, 1-np.exp(1-0)), CW/MAX_LATENCY, CH, fill=True, edgecolor='tab:blue'))
    ax.add_patch(patches.Ellipse((SLO_CPU_UTILIZATION, 1-np.exp(1-SLO_CPU_UTILIZATION)), CW/MAX_LATENCY, CH, zorder=10, fill=True, edgecolor='tab:blue', facecolor='white'))
    ax.add_patch(patches.Ellipse((SLO_CPU_UTILIZATION, 0), CW/MAX_LATENCY, CH, zorder=10, fill=True, edgecolor='tab:blue', facecolor='tab:blue'))
    ax.add_patch(patches.Ellipse((MAX_CPU_UTILIZATION, 0), CW/MAX_LATENCY, CH, zorder=10, fill=True, edgecolor='tab:blue', facecolor='tab:blue'))

    ax.set_xlabel('CPU Utilization')
    ax.set_ylabel('Reward')
    #ax.set_aspect("equal")
    ax.legend()
    ax.set_title('Cost reward')

    ax = fig.add_subplot(gs[:,1])
    """Plots a 2D heatmap of the total reward function."""
    latencies = np.linspace(0, MAX_LATENCY, 20)
    cpu_utilizations = np.linspace(0, MAX_CPU_UTILIZATION, 20)

    rewards = np.zeros((len(cpu_utilizations), len(latencies)))

    for i, cpu in enumerate(cpu_utilizations):
        for j, lat in enumerate(latencies):
            rewards[i, j] = total_reward(lat, cpu)

    mesh = ax.pcolormesh(latencies, cpu_utilizations, rewards, shading='auto', cmap='viridis')
    fig.colorbar(mesh, ax=ax)

    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('CPU Utilization')
    ax.set_title('Total Reward Heatmap')

    plt.tight_layout()
    plt.savefig('../results/reward_functions.pdf')
    plt.show()


plot_reward_functions()
