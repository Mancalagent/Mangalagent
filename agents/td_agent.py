from agents.base_agent import BaseAgent


class TDAgent(BaseAgent):
    def __init__(self, agent_id, network):
        super().__init__(agent_id)
        self.net = network
    def act(self, state):
        actions = self.get_available_actions(state)
        action = max(actions, key=lambda x: self.net(state)[x].item())
        return action