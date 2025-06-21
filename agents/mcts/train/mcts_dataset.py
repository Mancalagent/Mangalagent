import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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


class MCTSTrajectoryDataset(Dataset):
    """
    Dataset for trajectory-based MCTS training.
    Each sample is a complete trajectory (sequence of states, actions, values).
    """
    def __init__(self, trajectories):
        """
        Initialize with MCTS trajectories.
        
        Args:
            trajectories: List of trajectory lists from MCTS, where each trajectory contains
                          tuples of (state, action, value)
        """
        self.trajectories = []
        self._process_trajectories(trajectories)
    
    def _process_trajectories(self, trajectories):
        """Process raw trajectories into sequence format"""
        # Trajectories is a list of lists (one per game)
        for game_trajectories in trajectories:
            # Each game may have multiple trajectories (one per simulation)
            for trajectory in game_trajectories:
                if len(trajectory) == 0:  # Skip empty trajectories
                    continue
                    
                sequence_states = []
                sequence_actions = []
                sequence_values = []
                
                # Process each step in the trajectory
                for state, action, value in trajectory:
                    # Convert state to tensor
                    if hasattr(state, 'get_state'):
                        board_state = state.get_state()
                    else:
                        board_state = state
                    sequence_states.append(torch.FloatTensor(board_state))
                    
                    # Ensure action is valid integer
                    if isinstance(action, (int, np.integer)):
                        sequence_actions.append(int(action))
                    else:
                        raise ValueError(f"Expected integer action, got {type(action)}: {action}")
                    
                    sequence_values.append(float(value))
                
                # Store the complete trajectory
                self.trajectories.append({
                    'states': sequence_states,
                    'actions': sequence_actions,
                    'values': sequence_values,
                    'length': len(sequence_states)
                })
    
    def __len__(self):
        """Return the number of trajectories"""
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """Get a trajectory by index"""
        return self.trajectories[idx]


def trajectory_collate_fn(batch):
    """
    Custom collate function for trajectory batching.
    Handles variable-length sequences with padding and masking.
    
    Args:
        batch: List of trajectory dictionaries
        
    Returns:
        Dictionary with padded sequences and attention masks
    """
    if len(batch) == 0:
        return None
    
    # Extract sequences from batch
    states_batch = [torch.stack(item['states']) for item in batch]
    actions_batch = [torch.tensor(item['actions'], dtype=torch.int64) for item in batch]
    values_batch = [torch.tensor(item['values'], dtype=torch.float32) for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Pad sequences to same length
    max_length = max(lengths)
    
    # Pad states (batch_size, max_seq_len, state_dim)
    padded_states = pad_sequence(states_batch, batch_first=True, padding_value=0.0)
    
    # Pad actions (batch_size, max_seq_len)
    padded_actions = pad_sequence(actions_batch, batch_first=True, padding_value=0)
    
    # Pad values (batch_size, max_seq_len)
    padded_values = pad_sequence(values_batch, batch_first=True, padding_value=0.0)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    
    return {
        'states': padded_states,
        'actions': padded_actions,
        'values': padded_values,
        'attention_mask': attention_mask,
        'lengths': torch.tensor(lengths, dtype=torch.int32)
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


def create_mcts_trajectory_dataloader(trajectories, batch_size=32, shuffle=True):
    """
    Create a DataLoader for trajectory-based training.
    Each batch contains complete trajectories with padding and masking.
    
    Args:
        trajectories: List of trajectories from MCTS self-play
        batch_size: Number of trajectories per batch
        shuffle: Whether to shuffle the trajectories
        
    Returns:
        DataLoader object with trajectory batching
    """
    dataset = MCTSTrajectoryDataset(trajectories)
    
    # Handle empty dataset case
    if len(dataset) == 0:
        print("Warning: Empty trajectory dataset! No trajectories to train on.")
        # Return a dummy dataloader
        dummy_data = {
            'states': torch.zeros(1, 1, 14),
            'actions': torch.zeros(1, 1, dtype=torch.int64),
            'values': torch.zeros(1, 1, dtype=torch.float32),
            'attention_mask': torch.ones(1, 1, dtype=torch.bool),
            'lengths': torch.tensor([1], dtype=torch.int32)
        }
        return DataLoader([dummy_data], batch_size=1, shuffle=False)
    
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        collate_fn=trajectory_collate_fn,
        num_workers=2,
        pin_memory=True
    )