from agents.base_agent import BaseAgent

class MCTSAgent(BaseAgent):
    def __init__(self, id):
        super().__init__(id)
        
    def act(self, state):
        board, p_index = state
        self.player_index = p_index
        self.game_state = board
        
        # TODO: Implement MCTS
        
        
        