import random
from agents.base_agent import BaseAgent
from utils.util import Util

class RandomAgent(BaseAgent):
    """A simple agent that makes random valid moves"""
    
    def act(self, state):
        """Choose a random valid action
        
        Args:
            state: The current game state
        """
        board, p_index = state
        self.player_index = p_index
        self.game_state = board
        

        action = random.choice(self.get_available_actions(state))
        Util.save_board_state(self.id, board, action)
        # print(f"As player {self.player_index}, taking action {action}")
        return action
    
    