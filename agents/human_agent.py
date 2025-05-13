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
        
        

        # Always ask for input 0-5 regardless of player
        pit_index = int(input(f"Player {p_index}, choose a pit (0-5): "))
        print(f"Pit index: {pit_index}")
        valid_actions = self.get_available_actions(state)
        is_valid = False
        while not is_valid:
            pit_index = int(input(f"Choose a pit from {valid_actions}: "))
            if pit_index in valid_actions:
                is_valid = True
            else:
                print(f"Invalid choice. Choose a pit from {valid_actions}: ")

        Util.save_board_state(self.id, board, pit_index)
        print(f"As player {self.player_index}, taking action {pit_index}")
        
        return pit_index

    

    
    
