from agents.base_agent import BaseAgent
from utils.util import Util

class HumanAgent(BaseAgent):
    """Agent controlled by human input"""
    
    
    def __init__(self, id):
        super().__init__(id)
         

        
    def act(self, state):
        """Get action from human input
        
        Args:
            state: The current game state
        """
        board, p_index = state
        self.player_index = p_index
        self.game_state = board
        
        
        Util.save_board_state(self.id, board, p_index)
        # Always ask for input 0-5 regardless of player
        pit_index = int(input(f"Player {p_index}, choose a pit (0-5): "))
        print(f"Pit index: {pit_index}")
        while pit_index < 0 or pit_index > 5:
            # print(f"Invalid pit index, please pick one of these {self.get_available_actions(state)}")
            pit_index = int(input(f"Invalid choice. Choose a pit from 0-5: "))
            return pit_index  # Return 0-5, Mangala class will handle translation
        
        print(f"As player {self.player_index}, taking action {pit_index}")
        
        return pit_index

    

    
    
