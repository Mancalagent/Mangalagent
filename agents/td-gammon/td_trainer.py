import torch
from torch import optim

from agents.base_agent import BaseAgent
from mangala.mangala import Mangala
from . import TDNetwork


class TDTrainer:
    def __init__(self, game: Mangala, agent: BaseAgent, network, learning_rate=0.01, discount_factor=0.9,
                 trace_decay=0.7):
        self.game = game
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
        for episode in range(episodes):
            state = self.game.reset()
            done = False
            e_trace = [torch.zeros_like(p) for p in self.net.parameters()]
            while not self.game.game_over:
                action = self.agent.act(state)
                next_state, reward, is_terminal = self.game.transition(state, action)

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
                        p += self.learning_rate * td_error * e_trace[i]

                state = next_state

    def save_model(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load_model(self, filepath):
        self.net.load_state_dict(torch.load(filepath))

