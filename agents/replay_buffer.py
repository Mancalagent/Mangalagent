import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, is_terminal):
        self.buffer.append((state, action, reward, next_state, is_terminal))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None  # Not enough samples to form a batch
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*experiences)
        return (list(states), list(actions), list(rewards), list(next_states), list(terminals))

    def __len__(self):
        return len(self.buffer)
