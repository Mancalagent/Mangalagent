from agents.base_agent import BaseAgent
from mangala.mangala import Mangala


class TDAgent(BaseAgent):
    def __init__(self, agent_id, network):
        super().__init__(agent_id)
        self.net = network

    def act(self, state):
        board, p_index_ = state
        actions = self.get_available_actions(board)
        import torch

        action_values = []
        for action in actions:
            next_board, _, _ = Mangala.transition(board, action)
            value = self.net(next_board)
            action_values.append(value.item())
            # print(f"Action: {action}, Value: {value.item()}")

        action_probs = torch.softmax(torch.tensor(action_values), dim=0)
        # print(f"Action probabilities: {action_probs.tolist()}")

        max_action = actions[torch.argmax(action_probs).item()]
        return max_action
