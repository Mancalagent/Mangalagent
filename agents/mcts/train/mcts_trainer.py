from agents.mcts.mcts_tree import MCTSTree
from agents.mcts.mcts_network import MCTS_Policy_Network, MCTS_Value_Network
from agents.mcts.mcts_node import State
from tqdm.auto import tqdm

class MCTSTrainer:
    def __init__(self, **kwargs):
        
        # initial board state for the game
        board = [4] * 14
        board[6] = 0
        board[13] = 0
        
        self.game_count = kwargs.get('game_count', 1000)
        self.policy_network = kwargs.get('policy_network', MCTS_Policy_Network(input_dim=14, output_dim=6))
        self.value_network = kwargs.get('value_network', MCTS_Value_Network(input_dim=14, output_dim=1))
        self.tree = MCTSTree(init_state=State(board), policy_network=self.policy_network, value_network=self.value_network)
        
        
        
    def train(self, **kwargs):
        trajectories = []
        
        # Check if progress bar is requested
        use_progress_bar = kwargs.get('progress_bar', False)
        
        # Create a tqdm progress bar if requested
        game_iterator = tqdm(
            range(self.game_count), 
            desc="Generating self-play games",
            unit="game"
        ) if use_progress_bar else range(self.game_count)
        
        for game in game_iterator:
            iter_path = self.tree.run_iteration(c_puct=1.0)
            trajectories.append(iter_path)
            # print(iter_path)
            
            # Update progress bar if using one
            # if use_progress_bar:
            #     game_iterator.set_postfix({
            #         'trajectory_len': len(iter_path) if iter_path else 0
            #     })
            
        return trajectories    
            
            
            
            
           
           
           
        
        
        
        
        
        
        
        
        

