import pygame
import numpy as np

# Pygame setup
pygame.init()
width, height = 600, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Policy Selection with Uncertainty")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
BLUE = (50, 150, 255)
GRAY = (180, 180, 180)

# Grid environment
grid_size = 10
rows, cols = height // grid_size, width // grid_size
goal = np.array([cols - 2, rows - 2], dtype=float)
obstacles = [(15, 10), (16, 10), (17, 10), (17, 11), (17, 12)]

# Agent state
agent_pos = np.array([1, 1], dtype=float)
T = 100

# Policy horizon
H = 5

# Actions: up, down, left, right
actions = {
    "up": np.array([0, -1]),
    "down": np.array([0, 1]),
    "left": np.array([-1, 0]),
    "right": np.array([1, 0]),
}

def is_valid(pos):
    """Check if a position is valid (inside grid and not in obstacle)."""
    return (0 <= pos[0] < cols and 0 <= pos[1] < rows and tuple(pos.astype(int)) not in obstacles)

# Build policy bank (all sequences of length H)
def generate_policies(H):
    if H == 1:
        return [[a] for a in actions]
    smaller = generate_policies(H - 1)
    return [s + [a] for s in smaller for a in actions]

policy_bank = generate_policies(H)

# Likelihood: how close final state of policy is to current uncertain goal
def likelihood(policy, start, goal):
    pos = np.array(start)
    for act in policy:
        next_pos = pos + actions[act]

        # Add probabilistic transitions (slip)
        slip = np.random.choice([0, 1], p=[0.8, 0.2])
        if slip:
            # Random slip
            slip_act = np.random.choice(list(actions.keys()))
            next_pos = pos + actions[slip_act]

        if is_valid(next_pos):
            pos = next_pos
    dist = np.linalg.norm(pos - goal)
    return np.exp(-0.1 * dist)  # higher if closer to goal

# Prior: prefer smoother policies (less switching)
def prior(policy):
    switches = sum(policy[i] != policy[i + 1] for i in range(len(policy) - 1))
    return np.exp(-0.2 * switches)

# Posterior = prior * likelihood
def select_best_policy(pos, goal):
    scores = []
    for p in policy_bank:
        scores.append(prior(p) * likelihood(p, pos, goal))
    best = np.argmax(scores)
    return policy_bank[best]

# Main loop
step = 0
running = True
trace = [agent_pos.copy()]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Slightly drift the goal to introduce uncertainty
    goal += np.random.normal(0, 0.05, size=2)
    goal = np.clip(goal, [0, 0], [cols - 1, rows - 1])

    # Draw grid
    for x in range(0, width, grid_size):
        pygame.draw.line(screen, GRAY, (x, 0), (x, height))
    for y in range(0, height, grid_size):
        pygame.draw.line(screen, GRAY, (0, y), (width, y))

    # Draw goal
    pygame.draw.rect(screen, GREEN, (int(goal[0]) * grid_size, int(goal[1]) * grid_size, grid_size, grid_size))

    # Draw obstacles
    for ox, oy in obstacles:
        pygame.draw.rect(screen, GRAY, (ox * grid_size, oy * grid_size, grid_size, grid_size))

    # Select and apply best policy
    best_policy = select_best_policy(agent_pos, goal)
    action = best_policy[0]
    next_pos = agent_pos + actions[action]

    # Apply probabilistic slippage
    if np.random.rand() < 0.2:
        slip_act = np.random.choice(list(actions.keys()))
        next_pos = agent_pos + actions[slip_act]

    if is_valid(next_pos):
        agent_pos = next_pos
        trace.append(agent_pos.copy())

    # Draw trace
    for tx, ty in trace:
        pygame.draw.circle(screen, BLUE, (int(tx) * grid_size + grid_size // 2, int(ty) * grid_size + grid_size // 2), 3)

    # Draw agent
    pygame.draw.rect(screen, RED, (int(agent_pos[0]) * grid_size, int(agent_pos[1]) * grid_size, grid_size, grid_size))

    # Display info
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Step: {step}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(5)
    step += 1

    if step > T or (int(agent_pos[0]), int(agent_pos[1])) == (int(goal[0]), int(goal[1])):
        pygame.time.wait(1000)
        running = False

pygame.quit()
