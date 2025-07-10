import pprint
import random

import torch
from torch import optim

from agents.replay_buffer import ReplayBuffer
from mangala.mangala import Mangala
from agents.td_gammon.td_network import TDNetwork
from agents.minimax_agent import MinimaxAgent
import matplotlib.pyplot as plt

class TDTrainerMinimax:
    def __init__(self, agent, network, learning_rate=0.01, discount_factor=0.9,
                 trace_decay=0.7, minimax_depth=5):
        self.agent = agent
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        if not network:
            self.net = TDNetwork()
        else:
            self.net = network
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.minimax_agent = MinimaxAgent(id=99, max_depth=minimax_depth)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64

    def train(self, episodes):
        losses = []
        for episode in range(episodes):
            state = [4] * 14
            state[6] = 0
            state[13] = 0
            e_trace = [torch.zeros_like(p) for p in self.net.parameters()]
            done = False
            episode_loss = 0.0
            epsilon = 0.9 if episode < 0.8 * episodes else 0.1
            while not done:

                actions = self.agent.get_available_actions(state)

                # Epsilon-greedy: explore with random move, exploit with Minimax - more tests needed
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = self.minimax_agent.act((state, 0))

                next_state, reward, is_terminal = Mangala.transition(state, action)
                if is_terminal:
                    done = True

                v = self.net(state)

                if not Mangala.check_for_extra_turn(state, action):
                    next_state = Mangala.flip_board(next_state)

                v_next = self.net(next_state).detach()

                td_error = reward - v if is_terminal else reward + self.discount_factor * v_next - v

                self.optimizer.zero_grad()
                v.backward()
                with torch.no_grad():
                    for i, p in enumerate(self.net.parameters()):
                        e_trace[i] = self.discount_factor * self.trace_decay * e_trace[i] + p.grad
                        p.grad.copy_(td_error * e_trace[i])
                self.optimizer.step()

                episode_loss += td_error.item() ** 2
                state = next_state

            losses.append(episode_loss)
        pprint.pprint(losses)

    def buffer_train(self, episodes):
        losses = []
        for episode in range(episodes):
            state = [4] * 14
            state[6] = 0  # Player 0's store
            state[13] = 0  # Player 1's store

            # Reset eligibility trace for a new episode
            e_trace = [torch.zeros_like(p) for p in self.net.parameters()]

            done = False
            episode_loss = 0.0

            epsilon_start = 0.9
            epsilon_end = 0.1
            epsilon_decay_episodes = 0.8 * episodes
            epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay_episodes * (epsilon_start - epsilon_end))

            print(f"Episode {episode + 1}/{episodes}, Epsilon: {epsilon:.2f}")
            while not done:
                current_player_state = list(state)  # Player 0's perspective

                actions = self.agent.get_available_actions(current_player_state)
                if not actions:  # Handle no available moves
                    done = True
                    break

                # Epsilon-greedy: explore with random move, exploit with Minimax
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = self.minimax_agent.act((current_player_state, 0))  # Player 0's turn

                original_state_for_extra_turn_check = list(state)  # Snapshot before transition
                next_state, reward, is_terminal = Mangala.transition(state, action)

                state_for_buffer_next = list(next_state)  # Default to actual next_state

                if not Mangala.check_for_extra_turn(original_state_for_extra_turn_check, action):
                    state_for_buffer_next = Mangala.flip_board(
                        state_for_buffer_next)  # Flip for the opponent's perspective if turn changes

                self.replay_buffer.push(state, action, reward, state_for_buffer_next, is_terminal)

                # Update the network from a batch of experiences
                if len(self.replay_buffer) >= self.batch_size:
                    experiences = self.replay_buffer.sample(self.batch_size)
                    if experiences:
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = experiences

                        batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32)
                        batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32)
                        batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                        batch_terminals_tensor = torch.tensor(batch_terminals, dtype=torch.float32)  # Boolean to float

                        # Calculate V(s) for the batch
                        v_batch = self.net(batch_states_tensor).squeeze()

                        v_next_batch = self.net(batch_next_states_tensor).squeeze().detach()

                        # Calculate TD-Error for the batch
                        # Use torch.where to apply different logic for terminal states
                        td_target = torch.where(
                            batch_terminals_tensor == 1,
                            batch_rewards_tensor,
                            batch_rewards_tensor + self.discount_factor * v_next_batch
                        )
                        td_error_batch = td_target - v_batch

                        loss = (td_error_batch ** 2).mean()  # Mean squared error

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        episode_loss += loss.item()

                state = next_state
                if is_terminal:
                    done = True

            losses.append(episode_loss)
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid()
        plt.savefig('loss_plot.png')
        plt.close()
        pprint.pprint(losses)

    def save_model(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load_model(self, filepath):
        self.net.load_state_dict(torch.load(filepath))
