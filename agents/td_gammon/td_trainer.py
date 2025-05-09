import pprint

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)

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
            while not done:
                if flip:
                    state = Mangala.flip_board(state)
                    flip = False
                #print(f"Episode {episode}, State: {state}")
                actions = self.agent.get_available_actions(state)
                action = max(actions, key=lambda x: self.net(Mangala.transition(state, x)[0]).item())
                #print(f"Action: {action}")
                if Mangala.check_for_extra_turn(state,action):
                    flip = True

                next_state, reward, is_terminal = Mangala.transition(state, action)

                v = self.net(state)
                v_next = self.net(next_state)

                if is_terminal:
                    td_error = reward - v
                    done = True
                else:
                    td_error = reward + self.discount_factor * v_next - v

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
        plt.plot(losses)
        plt.xlabel("Episode")
        plt.ylabel("TD Error Squared")
        plt.title("TD Loss Over Episodes")
        plt.show()

    def save_model(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load_model(self, filepath):
        self.net.load_state_dict(torch.load(filepath))