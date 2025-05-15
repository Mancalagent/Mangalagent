from agents.base_agent import BaseAgent


class TDAgent(BaseAgent):
    def __init__(self, agent_id, network):
        super().__init__(agent_id)
        self.net = network
    def act(self, state):
        board,p_index_ = state
        actions = self.get_available_actions(board)
        action = max(actions, key=lambda a: self.net(board))
        #print(f"As player {p_index_}, taking action {action}")
        return action