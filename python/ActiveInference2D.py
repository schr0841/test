import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

# Set random seed
np.random.seed(4)

# Simulation parameters
T = 60
obs_noise_std = 1.0
motion_noise_std = 0.5

# Agent and belief state
agent_pos = np.zeros((T, 2))
belief_mu = np.zeros((T, 2))
belief_sigma2 = np.ones((T, 2)) * 10.0

# Target trajectory
target_pos = np.zeros((T, 2))
target_pos[0] = np.array([8.0, 8.0])

# Define dynamic drift (rotating)
def target_drift(t):
    angle = 0.2 * t
    return 0.1 * np.array([np.cos(angle), np.sin(angle)])

# Define action space
action_space = [
    np.array([0, 0]),
    np.array([-1, 0]), np.array([1, 0]),
    np.array([0, -1]), np.array([0, 1]),
    np.array([-1, -1]), np.array([-1, 1]),
    np.array([1, -1]), np.array([1, 1])
]

# Define expected free energy (EFE)
def expected_free_energy(mu, sigma2, agent_pos, target):
    dist_to_goal = np.sum((mu - target)**2)
    dist_from_agent = np.sum((mu - agent_pos)**2)
    uncertainty = np.sum(sigma2)
    return dist_to_goal + 0.3 * dist_from_agent + uncertainty

# Initialize
agent_pos[0] = np.array([0.0, 0.0])
belief_mu[0] = np.array([0.0, 0.0])

# Run simulation
for t in range(1, T):
    drift = target_drift(t)
    target_pos[t] = target_pos[t-1] + drift + np.random.normal(0, 0.02, size=2)

    obs = target_pos[t] + np.random.normal(0, obs_noise_std, size=2)

    K = belief_sigma2[t-1] / (belief_sigma2[t-1] + obs_noise_std**2)
    belief_mu[t] = belief_mu[t-1] + K * (obs - belief_mu[t-1])
    belief_sigma2[t] = (1 - K) * belief_sigma2[t-1]

    best_action = None
    best_cost = float("inf")
    for a in action_space:
        pred_pos = agent_pos[t-1] + a
        cost = expected_free_energy(belief_mu[t], belief_sigma2[t], pred_pos, target_pos[t])
        if cost < best_cost:
            best_cost = cost
            best_action = a

    agent_pos[t] = agent_pos[t-1] + best_action + np.random.normal(0, motion_noise_std, size=2)

# --------- Animation ---------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 14)
ax.set_ylim(-2, 14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Active Inference Agent with Drifting Target and Uncertainty Ellipses")
ax.grid(True)

agent_line, = ax.plot([], [], 'bo-', label="Agent Path")
belief_line, = ax.plot([], [], 'rx--', label="Belief Path")
target_line, = ax.plot([], [], 'y*', label="Target Path", markersize=12)
uncertainty_ellipse = Ellipse((0, 0), width=1, height=1, angle=0,
                              edgecolor='gray', facecolor='none', linestyle='--', label="Uncertainty")
ax.add_patch(uncertainty_ellipse)
ax.legend()

def init():
    agent_line.set_data([], [])
    belief_line.set_data([], [])
    target_line.set_data([], [])
    uncertainty_ellipse.set_visible(False)
    return agent_line, belief_line, target_line, uncertainty_ellipse

def animate(i):
    agent_line.set_data(agent_pos[:i+1, 0], agent_pos[:i+1, 1])
    belief_line.set_data(belief_mu[:i+1, 0], belief_mu[:i+1, 1])
    target_line.set_data(target_pos[:i+1, 0], target_pos[:i+1, 1])

    width = 2 * np.sqrt(belief_sigma2[i, 0])
    height = 2 * np.sqrt(belief_sigma2[i, 1])
    uncertainty_ellipse.set_center(belief_mu[i])
    uncertainty_ellipse.set_width(width)
    uncertainty_ellipse.set_height(height)
    uncertainty_ellipse.set_visible(True)

    return agent_line, belief_line, target_line, uncertainty_ellipse

ani = animation.FuncAnimation(fig, animate, frames=T, init_func=init,
                              blit=True, interval=200, repeat=False)

plt.show()