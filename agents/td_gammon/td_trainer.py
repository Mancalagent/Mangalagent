import pprint
import random

import matplotlib
import torch

matplotlib.use('TkAgg')
from torch import optim

from mangala.mangala import Mangala
from . import TDNetwork


class TDTrainer:
    def __init__(self, agent, network, learning_rate=0.01, discount_factor=0.9,
                 trace_decay=0.7):
        self.agent = agent
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        if not network:
            self.net = TDNetwork()
        else:
            self.net = network
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def train(self, episodes):
        losses = []
        for episode in range(episodes):
            state = [4]*14
            state[6] = 0
            state[13] = 0
            e_trace = [torch.zeros_like(p) for p in self.net.parameters()]
            flip = False
            done = False
            episode_loss = 0.0
            epsilon = max(0.1, 0.6 - episode / episodes)
            while not done:
                if flip:
                    state = Mangala.flip_board(state)
                    flip = False

                actions = self.agent.get_available_actions(state)

                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = max(actions, key=lambda x: self.net(Mangala.transition(state, x)[0]).item())


                if Mangala.check_for_extra_turn(state,action):
                    flip = True

                next_state, reward, is_terminal = Mangala.transition(state, action)
                if is_terminal:
                    done = True

                v = self.net(state)
                v_next = self.net(next_state)

                td_error = reward - v if is_terminal else reward + self.discount_factor * v_next - v

                self.optimizer.zero_grad()
                v.backward()
                with torch.no_grad():
                    for i, p in enumerate(self.net.parameters()):
                        grad = p.grad.detach()
                        e_trace[i] = self.discount_factor * self.trace_decay * e_trace[i] + grad
                        p_new = p + self.learning_rate * td_error * e_trace[i]
                        p.copy_(p_new)

                episode_loss += td_error.item() ** 2
                state = next_state
            losses.append(episode_loss)
        pprint.pprint(losses)

    def save_model(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load_model(self, filepath):
        self.net.load_state_dict(torch.load(filepath))