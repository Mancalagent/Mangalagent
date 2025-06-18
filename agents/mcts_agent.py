from agents.base_agent import BaseAgent
from agents.mcts.mcts_network import MCTS_Policy_Network
import torch
import numpy as np

class MCTSAgent(BaseAgent):
    def __init__(self, id, model_path = None):
        super().__init__(id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MCTS_Policy_Network()
        if model_path is not None:
            self.network.load_state_dict(torch.load(model_path))
        
    def act(self, state):
        board, p_index = state
        self.player_index = p_index
        self.game_state = board
        self.network.to(self.device)
        self.network.eval()
        
        with torch.no_grad():
            state_tensor = torch.tensor(self.game_state,dtype=torch.float32).to(self.device)
            policy = self.network.inference(state_tensor)
            policy = policy.cpu().numpy()
            try:
                print(f"MCTSAgent {self.id} policy: {policy}")
                action = np.argmax(policy)
                print(f"MCTSAgent {self.id} action: {action}")
            except Exception as e:
                print(f"MCTSAgent {self.id} policy: {policy}")
                print(f"Error: {e}")
                # Try to get the second max
                policy_copy = policy.copy()
                max_idx = np.argmax(policy_copy)
                policy_copy[max_idx] = -np.inf  # Exclude the max value
                action = np.argmax(policy_copy)
                
                print(f"MCTSAgent {self.id} second max action: {action}")
                return action
                
                
               
            return action
        
        
       
        
        
        
        