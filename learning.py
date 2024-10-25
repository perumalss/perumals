import numpy as np
import random

# Parameters
grid_size = 5
goal = (4, 4)
obstacles = [(2, 2), (3, 3)]
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, 4))  # Actions: up, down, left, right

# Actions
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

def is_valid(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size and state not in obstacles

def get_next_state(state, action):
    x, y = state
    dx, dy = actions[action]
    new_state = (x + dx, y + dy)
    return new_state if is_valid(new_state) else state

def get_reward(state):
    return 10 if state == goal else -1

# Q-learning algorithm
for episode in range(1000):
    state = (0, 0)  # Start position
    while state != goal:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(4))  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploit
        
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Update Q-value
        old_q_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_q_value + alpha * (reward + gamma * next_max - old_q_value)

        state = next_state

# Q-table learned
print("Learned Q-table:\n", q_table)
