#!/usr/bin/env python3
"""
Trajectory Manager for MCTS Training

This script provides utilities for generating, saving, loading, and analyzing
MCTS trajectories independently of the training process.
"""

import os
import argparse
import pickle
import json
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
from agents.mcts.train.mcts_trainer import MCTSTrainer

def format_trajectory_filename(filepath, num_games=None):
    """
    Format trajectory filename with automatic .pkl extension and datetime.
    
    Args:
        filepath: Base filepath (can be with or without .pkl extension)
        num_games: Number of games (optional, adds to filename)
        
    Returns:
        Formatted filepath with datetime and .pkl extension
    """
    # Remove .pkl extension if present
    if filepath.endswith('.pkl'):
        base_path = filepath[:-4]
    else:
        base_path = filepath
    
    # Add datetime stamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add number of games if provided
    if num_games is not None:
        formatted_path = f"{base_path}_{num_games}games_{timestamp}.pkl"
    else:
        formatted_path = f"{base_path}_{timestamp}.pkl"
    
    return formatted_path

def generate_trajectories(num_games, save_path=None, progress_bar=True):
    """
    Generate MCTS trajectories.
    
    Args:
        num_games: Number of games to generate
        save_path: Optional path to save trajectories (will auto-format with datetime)
        progress_bar: Whether to show progress bar
        
    Returns:
        List of trajectories
    """
    print(f"Generating {num_games} MCTS games...")
    
    mcts_trainer = MCTSTrainer(game_count=num_games)
    trajectories = mcts_trainer.train(progress_bar=progress_bar)
    
    print(f"Generated {len(trajectories)} game trajectories")
    
    if save_path:
        # Auto-format the filename with datetime and games count
        formatted_path = format_trajectory_filename(save_path, num_games)
        save_trajectories(trajectories, formatted_path)
    
    return trajectories

def save_trajectories(trajectories, filepath):
    """
    Save trajectories to disk with metadata.
    """
    print(f"Saving {len(trajectories)} trajectories to {filepath}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save as pickle
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    # Calculate and save metadata
    total_trajectories = sum(len(game_traj) for game_traj in trajectories)
    total_samples = sum(len(traj) for game_traj in trajectories for traj in game_traj)
    
    metadata = {
        'num_games': len(trajectories),
        'total_trajectories': total_trajectories,
        'total_samples': total_samples,
        'avg_trajectories_per_game': total_trajectories / len(trajectories),
        'avg_samples_per_trajectory': total_samples / total_trajectories if total_trajectories > 0 else 0,
        'timestamp': datetime.now().isoformat(),
        'filepath': filepath
    }
    
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Trajectories saved to {filepath}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total samples: {total_samples}")
    
    return metadata

def load_trajectories(filepath):
    """
    Load trajectories from disk.
    """
    # Auto-add .pkl extension if not present
    if not filepath.endswith('.pkl'):
        filepath = filepath + '.pkl'
    
    print(f"Loading trajectories from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} game trajectories")
    
    # Load metadata if available
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Total samples: {metadata['total_samples']}")
    
    return trajectories

def analyze_trajectories(trajectories, detailed=False):
    """
    Analyze trajectory statistics.
    """
    print("\n=== TRAJECTORY ANALYSIS ===")
    
    num_games = len(trajectories)
    total_trajectories = sum(len(game_traj) for game_traj in trajectories)
    total_samples = sum(len(traj) for game_traj in trajectories for traj in game_traj)
    
    print(f"Number of games: {num_games}")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Total samples: {total_samples}")
    print(f"Avg trajectories per game: {total_trajectories/num_games:.2f}")
    print(f"Avg samples per trajectory: {total_samples/total_trajectories:.2f}" if total_trajectories > 0 else "No trajectories")
    
    if detailed and trajectories:
        # Analyze trajectory lengths
        traj_lengths = []
        for game_traj in trajectories:
            for traj in game_traj:
                traj_lengths.append(len(traj))
        
        if traj_lengths:
            print(f"\nTrajectory length statistics:")
            print(f"  Min length: {min(traj_lengths)}")
            print(f"  Max length: {max(traj_lengths)}")
            print(f"  Avg length: {np.mean(traj_lengths):.2f}")
            print(f"  Std length: {np.std(traj_lengths):.2f}")
        
        # Analyze actions
        actions = []
        values = []
        for game_traj in trajectories:
            for traj in game_traj:
                for state, action, value in traj:
                    actions.append(action)
                    values.append(value)
        
        if actions:
            print(f"\nAction statistics:")
            print(f"  Action range: {min(actions)} to {max(actions)}")
            print(f"  Action distribution: {np.bincount(actions)}")
            
        if values:
            print(f"\nValue statistics:")
            print(f"  Value range: {min(values):.3f} to {max(values):.3f}")
            print(f"  Avg value: {np.mean(values):.3f}")
            print(f"  Value distribution: {np.bincount([int(v) for v in values])}")

def merge_trajectories(filepaths, output_path):
    """
    Merge multiple trajectory files into one.
    """
    print(f"Merging {len(filepaths)} trajectory files...")
    
    all_trajectories = []
    total_games = 0
    for filepath in filepaths:
        trajectories = load_trajectories(filepath)
        all_trajectories.extend(trajectories)
        total_games += len(trajectories)
        print(f"  Added {len(trajectories)} games from {filepath}")
    
    # Auto-format output filename with total games and datetime
    formatted_output = format_trajectory_filename(output_path, total_games)
    save_trajectories(all_trajectories, formatted_output)
    print(f"Merged trajectories saved to {formatted_output}")
    
    return all_trajectories

def main():
    parser = argparse.ArgumentParser(description="MCTS Trajectory Manager")
    parser.add_argument("command", choices=["generate", "analyze", "merge"], 
                       help="Command to execute")
    
    # Generation args
    parser.add_argument("--num-games", type=int, default=100, 
                       help="Number of games to generate")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path (auto-adds .pkl extension and datetime)")
    
    # Analysis args
    parser.add_argument("--input", type=str, help="Input trajectory file (auto-adds .pkl if needed)")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed analysis")
    
    # Merge args
    parser.add_argument("--inputs", type=str, nargs="+", 
                       help="Input trajectory files to merge (auto-adds .pkl if needed)")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        trajectories = generate_trajectories(
            num_games=args.num_games,
            save_path=args.output,
            progress_bar=True
        )
        analyze_trajectories(trajectories, detailed=args.detailed)
        
    elif args.command == "analyze":
        if not args.input:
            print("Error: --input required for analyze command")
            return
        trajectories = load_trajectories(args.input)
        analyze_trajectories(trajectories, detailed=args.detailed)
        
    elif args.command == "merge":
        if not args.inputs:
            print("Error: --inputs required for merge command")
            return
        trajectories = merge_trajectories(args.inputs, args.output)
        analyze_trajectories(trajectories, detailed=args.detailed)

if __name__ == "__main__":
    main()





# basic call 
# python agent/mcts/train/trajectory_manager.py generate --num-games 50 --output data/test_trajectories



"""
=== EXAMPLE RUN COMMANDS FOR TRAJECTORY MANAGER ===

# Basic Trajectory Generation:
# -----------------------------

# 1. Generate small dataset for testing (auto-adds datetime: test_50games_20241215_143022.pkl)
python trajectory_manager.py generate --num-games 50 --output "data/test_trajectories"

# 2. Generate medium dataset with detailed analysis
python trajectory_manager.py generate --num-games 500 --output "data/medium_trajectories" --detailed

# 3. Generate large dataset for production
python trajectory_manager.py generate --num-games 5000 --output "data/production_trajectories" --detailed

# 4. Generate multiple datasets for different experiments
python trajectory_manager.py generate --num-games 1000 --output "experiments/exp1_trajectories"
python trajectory_manager.py generate --num-games 1000 --output "experiments/exp2_trajectories"
python trajectory_manager.py generate --num-games 1000 --output "experiments/exp3_trajectories"


# Trajectory Analysis:
# --------------------

# 5. Basic analysis of existing trajectories (auto-adds .pkl if needed)
python trajectory_manager.py analyze --input "data/test_50games_20241215_143022"

# 6. Detailed analysis with statistics
python trajectory_manager.py analyze --input "data/medium_500games_20241215_143125" --detailed

# 7. Analyze production dataset
python trajectory_manager.py analyze --input "data/production_5000games_20241215_144030" --detailed

# 8. Quick check of trajectory quality
python trajectory_manager.py analyze --input "experiments/exp1_1000games_20241215_145012"


# Merging Trajectories:
# ----------------------

# 9. Merge experimental datasets (output auto-formatted: merged_experiments_3000games_20241215_150045.pkl)
python trajectory_manager.py merge --inputs "experiments/exp1_1000games_20241215_145012" "experiments/exp2_1000games_20241215_145112" "experiments/exp3_1000games_20241215_145212" --output "data/merged_experiments"

# 10. Combine test and training sets
python trajectory_manager.py merge --inputs "data/test_50games_20241215_143022" "data/medium_500games_20241215_143125" --output "data/combined_dataset" --detailed

# 11. Create massive dataset from multiple sources
python trajectory_manager.py merge --inputs "data/production_5000games_20241215_144030" "data/merged_experiments_3000games_20241215_150045" --output "data/massive_dataset" --detailed


# Research Workflow Examples:
# ----------------------------

# 12. Generate datasets for hyperparameter search
python trajectory_manager.py generate --num-games 2000 --output "research/base_trajectories.pkl" --detailed

# Analysis before training
python trajectory_manager.py analyze --input "research/base_trajectories.pkl" --detailed

# 13. Create ablation study datasets
python trajectory_manager.py generate --num-games 1000 --output "ablation/dataset_v1.pkl"
python trajectory_manager.py generate --num-games 1000 --output "ablation/dataset_v2.pkl"
python trajectory_manager.py generate --num-games 1000 --output "ablation/dataset_v3.pkl"

# Merge for final comparison
python trajectory_manager.py merge --inputs "ablation/dataset_v1.pkl" "ablation/dataset_v2.pkl" "ablation/dataset_v3.pkl" --output "ablation/final_dataset.pkl" --detailed


# Quality Control and Validation:
# --------------------------------

# 14. Generate validation set
python trajectory_manager.py generate --num-games 200 --output "validation/validation_trajectories.pkl" --detailed

# 15. Check data quality before training
python trajectory_manager.py analyze --input "data/production_trajectories.pkl" --detailed

# 16. Verify merged dataset integrity
python trajectory_manager.py analyze --input "data/massive_dataset.pkl" --detailed


# Incremental Dataset Building:
# ------------------------------

# 17. Build dataset incrementally (for long-running experiments)
python trajectory_manager.py generate --num-games 1000 --output "incremental/batch_1.pkl"
python trajectory_manager.py generate --num-games 1000 --output "incremental/batch_2.pkl"
python trajectory_manager.py generate --num-games 1000 --output "incremental/batch_3.pkl"

# Combine incremental batches
python trajectory_manager.py merge --inputs "incremental/batch_1.pkl" "incremental/batch_2.pkl" "incremental/batch_3.pkl" --output "incremental/combined_3k.pkl" --detailed

# 18. Add more data to existing dataset
python trajectory_manager.py generate --num-games 2000 --output "incremental/batch_4.pkl"
python trajectory_manager.py merge --inputs "incremental/combined_3k.pkl" "incremental/batch_4.pkl" --output "incremental/final_5k.pkl" --detailed


# Performance Testing:
# ---------------------

# 19. Generate different sizes for performance comparison
python trajectory_manager.py generate --num-games 100 --output "perf_test/small.pkl" --detailed
python trajectory_manager.py generate --num-games 1000 --output "perf_test/medium.pkl" --detailed
python trajectory_manager.py generate --num-games 10000 --output "perf_test/large.pkl" --detailed

# Analyze each for comparison
python trajectory_manager.py analyze --input "perf_test/small.pkl" --detailed
python trajectory_manager.py analyze --input "perf_test/medium.pkl" --detailed
python trajectory_manager.py analyze --input "perf_test/large.pkl" --detailed


# Production Pipeline:
# ---------------------

# 20. Complete production data pipeline
# Step 1: Generate base dataset
python trajectory_manager.py generate --num-games 8000 --output "production/base_8k.pkl" --detailed

# Step 2: Generate additional validation data
python trajectory_manager.py generate --num-games 1000 --output "production/validation_1k.pkl" --detailed

# Step 3: Generate test data
python trajectory_manager.py generate --num-games 1000 --output "production/test_1k.pkl" --detailed

# Step 4: Verify all datasets
python trajectory_manager.py analyze --input "production/base_8k.pkl" --detailed
python trajectory_manager.py analyze --input "production/validation_1k.pkl" --detailed
python trajectory_manager.py analyze --input "production/test_1k.pkl" --detailed

# Step 5: Create combined training set (optional)
python trajectory_manager.py merge --inputs "production/base_8k.pkl" "production/validation_1k.pkl" --output "production/training_9k.pkl" --detailed


# Debugging and Development:
# ---------------------------

# 21. Quick debug dataset
python trajectory_manager.py generate --num-games 10 --output "debug/tiny.pkl" --detailed

# 22. Check if specific file has issues
python trajectory_manager.py analyze --input "debug/tiny.pkl" --detailed

# 23. Test merging functionality
python trajectory_manager.py generate --num-games 5 --output "debug/test1.pkl"
python trajectory_manager.py generate --num-games 5 --output "debug/test2.pkl"
python trajectory_manager.py merge --inputs "debug/test1.pkl" "debug/test2.pkl" --output "debug/merged.pkl" --detailed


# Usage with train.py Integration:
# ---------------------------------

# 24. Full workflow: Generate -> Analyze -> Train
python trajectory_manager.py generate --num-games 2000 --output "workflow/trajectories.pkl" --detailed
python trajectory_manager.py analyze --input "workflow/trajectories.pkl" --detailed
python train.py --load-trajectories --trajectories-file "workflow/trajectories.pkl" --epochs 30 --output-dir "workflow/models"

# 25. Multi-experiment workflow
python trajectory_manager.py generate --num-games 3000 --output "multi_exp/shared_data.pkl" --detailed

# Run multiple experiments with same data
python train.py --load-trajectories --trajectories-file "multi_exp/shared_data.pkl" --value-update-freq 1 --policy-update-freq 1 --output-dir "multi_exp/exp1"
python train.py --load-trajectories --trajectories-file "multi_exp/shared_data.pkl" --value-update-freq 2 --policy-update-freq 1 --output-dir "multi_exp/exp2"
python train.py --load-trajectories --trajectories-file "multi_exp/shared_data.pkl" --value-update-freq 1 --policy-update-freq 2 --output-dir "multi_exp/exp3"
""" 