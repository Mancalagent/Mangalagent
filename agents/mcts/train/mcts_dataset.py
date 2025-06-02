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
            for trajectory in game_trajectories:
                # Each trajectory has tuples of (state, action, value)
                for state, action, value in trajectory:
                    # Convert state to tensor (assuming state is board state)
                    if hasattr(state, 'get_state'):
                        board_state = state.get_state()
                    else:
                        board_state = state
                    self.states.append(torch.FloatTensor(board_state))
                    
                    policy = action                   


                    self.policy_targets.append(policy)
                    
                    # Store value target
                    self.value_targets.append(float(value))
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.states)
    
    def __getitem__(self, idx):
        """Get a sample by index"""
        return {
            'state': self.states[idx],
            'policy_target': self.policy_targets[idx],
            'value_target': self.value_targets[idx]
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
        num_workers=2,  # Reduced worker count for debugging
        pin_memory=True
    )