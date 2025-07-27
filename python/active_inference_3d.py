
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Set seed
np.random.seed(42)

# Time steps
T = 60

# Initialize state arrays
agent_pos = np.zeros((T, 3))
belief_mu = np.zeros((T, 3))
belief_sigma2 = np.ones((T, 3)) * 10.0
target_pos = np.zeros((T, 3))
target_pos[0] = np.array([8.0, 8.0, 8.0])

# Noise parameters
obs_noise_std = 1.0
motion_noise_std = 0.5

# Action space (27 discrete movements in 3D)
action_space = [np.array([dx, dy, dz])
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]]

# Target drift
def target_drift(t):
    return 0.1 * np.array([
        np.cos(0.1 * t),
        np.sin(0.1 * t),
        np.sin(0.05 * t)
    ])

# Expected free energy
def expected_free_energy(mu, sigma2, agent_pos, target):
    dist_to_goal = np.sum((mu - target)**2)
    dist_from_agent = np.sum((mu - agent_pos)**2)
    uncertainty = np.sum(sigma2)
    return dist_to_goal + 0.3 * dist_from_agent + uncertainty

# Init agent
agent_pos[0] = np.array([0.0, 0.0, 0.0])
belief_mu[0] = np.array([0.0, 0.0, 0.0])

# Simulation loop
for t in range(1, T):
    drift = target_drift(t)
    target_pos[t] = target_pos[t-1] + drift + np.random.normal(0, 0.02, size=3)

    obs = target_pos[t] + np.random.normal(0, obs_noise_std, size=3)

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

    noise = np.random.normal(0, motion_noise_std, size=3)
    agent_pos[t] = agent_pos[t-1] + best_action + noise

# -------------- 3D Animation --------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 14)
ax.set_ylim(-2, 14)
ax.set_zlim(-2, 14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Active Inference Agent with Drifting Target")

agent_line, = ax.plot([], [], [], 'bo-', label="Agent")
belief_line, = ax.plot([], [], [], 'rx--', label="Belief")
target_line, = ax.plot([], [], [], 'y*', label="Target", markersize=10)

ax.legend()

def init():
    agent_line.set_data([], [])
    agent_line.set_3d_properties([])
    belief_line.set_data([], [])
    belief_line.set_3d_properties([])
    target_line.set_data([], [])
    target_line.set_3d_properties([])
    return agent_line, belief_line, target_line

def update(i):
    agent_line.set_data(agent_pos[:i+1, 0], agent_pos[:i+1, 1])
    agent_line.set_3d_properties(agent_pos[:i+1, 2])

    belief_line.set_data(belief_mu[:i+1, 0], belief_mu[:i+1, 1])
    belief_line.set_3d_properties(belief_mu[:i+1, 2])

    target_line.set_data(target_pos[:i+1, 0], target_pos[:i+1, 1])
    target_line.set_3d_properties(target_pos[:i+1, 2])

    return agent_line, belief_line, target_line

ani = animation.FuncAnimation(fig, update, frames=T, init_func=init,
                              blit=False, interval=200, repeat=False)

plt.tight_layout()
plt.show()
