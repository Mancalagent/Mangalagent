#!/usr/bin/env python3
"""
Simple test script to verify trajectory-based training functionality.
This script creates dummy trajectories and tests both dataset classes.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.mcts.train.mcts_dataset import (
    MCTSDataset, 
    MCTSTrajectoryDataset, 
    create_mcts_dataloader, 
    create_mcts_trajectory_dataloader,
    trajectory_collate_fn
)

def create_dummy_trajectories(num_games=3, max_traj_per_game=2, max_steps_per_traj=5):
    """Create dummy trajectories for testing"""
    trajectories = []
    
    for game_idx in range(num_games):
        game_trajectories = []
        num_trajs = np.random.randint(1, max_traj_per_game + 1)
        
        for traj_idx in range(num_trajs):
            trajectory = []
            num_steps = np.random.randint(2, max_steps_per_traj + 1)
            
            for step in range(num_steps):
                # Create dummy state (14-dimensional for Mangala)
                state = np.random.rand(14).astype(np.float32)
                # Random action (0-5 for Mangala)
                action = np.random.randint(0, 6)
                # Random value (-1 to 1)
                value = np.random.uniform(-1, 1)
                
                trajectory.append((state, action, value))
            
            game_trajectories.append(trajectory)
        trajectories.append(game_trajectories)
    
    return trajectories

def test_standard_dataset():
    """Test the standard MCTSDataset"""
    print("=" * 50)
    print("Testing Standard MCTSDataset")
    print("=" * 50)
    
    # Create dummy data
    trajectories = create_dummy_trajectories(num_games=3, max_traj_per_game=2, max_steps_per_traj=4)
    
    # Test dataset
    dataset = MCTSDataset(trajectories)
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  State shape: {sample['state'].shape}")
        print(f"  Policy target: {sample['policy_target']}")
        print(f"  Value target: {sample['value_target']}")
    
    # Test dataloader
    dataloader = create_mcts_dataloader(trajectories, batch_size=4, shuffle=True)
    print(f"\nTesting dataloader with batch_size=4:")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  States shape: {batch['state'].shape}")
        print(f"  Policy targets shape: {batch['policy_target'].shape}")
        print(f"  Value targets shape: {batch['value_target'].shape}")
        if batch_idx >= 2:  # Only test first few batches
            break

def test_trajectory_dataset():
    """Test the trajectory-based MCTSTrajectoryDataset"""
    print("\n" + "=" * 50)
    print("Testing Trajectory MCTSTrajectoryDataset")
    print("=" * 50)
    
    # Create dummy data
    trajectories = create_dummy_trajectories(num_games=4, max_traj_per_game=3, max_steps_per_traj=6)
    
    # Test dataset
    dataset = MCTSTrajectoryDataset(trajectories)
    print(f"Dataset length (number of trajectories): {len(dataset)}")
    
    # Test a few trajectory samples
    for i in range(min(3, len(dataset))):
        traj = dataset[i]
        print(f"Trajectory {i}:")
        print(f"  Length: {traj['length']}")
        print(f"  States: {len(traj['states'])} states of shape {traj['states'][0].shape if traj['states'] else 'N/A'}")
        print(f"  Actions: {traj['actions']}")
        print(f"  Values: {traj['values']}")
    
    # Test dataloader with collate function
    dataloader = create_mcts_trajectory_dataloader(trajectories, batch_size=3, shuffle=True)
    print(f"\nTesting trajectory dataloader with batch_size=3:")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  States shape: {batch['states'].shape}")  # (batch_size, max_seq_len, state_dim)
        print(f"  Actions shape: {batch['actions'].shape}")  # (batch_size, max_seq_len)
        print(f"  Values shape: {batch['values'].shape}")  # (batch_size, max_seq_len)
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Lengths: {batch['lengths']}")
        print(f"  Attention mask sample: {batch['attention_mask'][0]}")
        if batch_idx >= 1:  # Only test first few batches
            break

def test_collate_function():
    """Test the trajectory collate function directly"""
    print("\n" + "=" * 50)
    print("Testing Trajectory Collate Function")
    print("=" * 50)
    
    # Create manual batch data
    batch_data = []
    
    # Trajectory 1: 3 steps
    traj1 = {
        'states': [torch.randn(14) for _ in range(3)],
        'actions': [0, 2, 4],
        'values': [0.5, -0.2, 0.8],
        'length': 3
    }
    
    # Trajectory 2: 5 steps
    traj2 = {
        'states': [torch.randn(14) for _ in range(5)],
        'actions': [1, 3, 0, 5, 2],
        'values': [-0.1, 0.3, -0.5, 0.7, 0.2],
        'length': 5
    }
    
    # Trajectory 3: 2 steps
    traj3 = {
        'states': [torch.randn(14) for _ in range(2)],
        'actions': [4, 1],
        'values': [0.1, -0.3],
        'length': 2
    }
    
    batch_data = [traj1, traj2, traj3]
    
    # Test collate function
    collated = trajectory_collate_fn(batch_data)
    
    print("Collated batch:")
    print(f"  States shape: {collated['states'].shape}")
    print(f"  Actions shape: {collated['actions'].shape}")
    print(f"  Values shape: {collated['values'].shape}")
    print(f"  Attention mask shape: {collated['attention_mask'].shape}")
    print(f"  Lengths: {collated['lengths']}")
    
    print("\nDetailed attention mask:")
    for i, mask in enumerate(collated['attention_mask']):
        print(f"  Trajectory {i}: {mask} (length: {collated['lengths'][i]})")

def main():
    """Run all tests"""
    print("Testing MCTS Trajectory-based Training Components")
    print("=" * 70)
    
    try:
        test_standard_dataset()
        test_trajectory_dataset()
        test_collate_function()
        
        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("The trajectory-based training components are working correctly.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 