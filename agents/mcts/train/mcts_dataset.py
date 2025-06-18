import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MCTSDataset(Dataset):
    """
    Custom dataset for MCTS training data.
    
    Stores game states, policy targets (action probabilities), and value targets.
    """
    def __init__(self, trajectories):
        """
        Initialize the dataset with MCTS trajectories.
        
        Args:
            trajectories: List of trajectory lists from MCTS, where each trajectory contains
                          tuples of (state, action, value)
        """
        self.states = []
        self.policy_targets = []
        self.value_targets = []
        
        self._process_trajectories(trajectories)
        
        
    
    def _process_trajectories(self, trajectories):
        """Process raw trajectories into training samples"""
        # Trajectories is a list of lists (one per game)
        for game_trajectories in trajectories:
            # Each game may have multiple trajectories (one per simulation)
            # print(f"processing game {game_trajectories}")
            for trajectory in game_trajectories:
                # print(f"processing trajectory {trajectory}")
                # Each trajectory has tuples of (state, action, value)
                for state, action, value in trajectory:
                    # print(f"processing state {state}")
                    # Convert state to tensor (assuming state is board state)
                    if hasattr(state, 'get_state'):
                        board_state = state.get_state()
                    else:
                        board_state = state
                    self.states.append(torch.FloatTensor(board_state))
                    
                    # Action should be an integer index (0-5 for Mangala)
                    # Ensure it's a valid integer action
                    if isinstance(action, (int, np.integer)):
                        policy_target = int(action)
                    else:
                        raise ValueError(f"Expected integer action, got {type(action)}: {action}")
                    
                    self.policy_targets.append(policy_target)
                    
                    # Store value target
                    self.value_targets.append(float(value))
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.states)
    
    def __getitem__(self, idx):
        """Get a sample by index"""
        return {
        "state"        : torch.as_tensor(self.states[idx], dtype=torch.float32),
        "policy_target": torch.tensor(self.policy_targets[idx], dtype=torch.int64),
        "value_target" : torch.tensor(self.value_targets[idx], dtype=torch.float32),
    }


def create_mcts_dataloader(trajectories, batch_size=64, shuffle=True):
    """
    Create a DataLoader from MCTS trajectories.
    
    Args:
        trajectories: List of trajectories from MCTS self-play
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader object
    """
    dataset = MCTSDataset(trajectories)
    
    # Handle empty dataset case
    if len(dataset) == 0:
        print("Warning: Empty dataset! No trajectories to train on.")
        # Return a dummy dataloader with a single zero sample
        dummy_data = {
            'state': torch.zeros(14),
            'policy_target': torch.zeros(6),
            'value_target': torch.tensor(0.0)
        }
        return DataLoader([dummy_data], batch_size=1, shuffle=False)
    
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),  # Ensure batch size isn't larger than dataset
        shuffle=shuffle,
        num_workers=2,  
        pin_memory=True
    )