import torch
import os
import argparse
import pickle
import json
from datetime import datetime
from tqdm.auto import tqdm
from agents.mcts.train.mcts_trainer import MCTSTrainer
from agents.mcts.train.network_trainer import NetworkTrainer

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

def save_trajectories(trajectories, filepath):
    """
    Save trajectories to disk.
    
    Args:
        trajectories: List of trajectory data from MCTS
        filepath: Path to save the trajectories file (will auto-format)
    """
    print(f"Saving {len(trajectories)} trajectories to {filepath}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save as pickle for Python objects
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"Trajectories saved successfully!")
    
    # Also save metadata as JSON for easy inspection
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    metadata = {
        'num_games': len(trajectories),
        'total_trajectories': sum(len(game_traj) for game_traj in trajectories),
        'total_samples': sum(len(traj) for game_traj in trajectories for traj in game_traj),
        'timestamp': datetime.now().isoformat(),
        'filepath': filepath
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    return metadata

def load_trajectories(filepath):
    """
    Load trajectories from disk.
    
    Args:
        filepath: Path to the trajectories file
        
    Returns:
        List of trajectory data
    """
    print(f"Loading trajectories from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectories file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} game trajectories successfully!")
    
    # Load and display metadata if available
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata: {metadata['total_samples']} total samples from {metadata['num_games']} games")
    
    return trajectories

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    trajectories = None
    
    # Determine trajectory file path
    if args.trajectories_file:
        trajectories_path = args.trajectories_file
    else:
        trajectories_path = format_trajectory_filename(os.path.join(args.output_dir, "trajectories"), args.mcts_games)
    
    # Load existing trajectories or generate new ones
    if args.load_trajectories and os.path.exists(trajectories_path):
        trajectories = load_trajectories(trajectories_path)
        print("loaded trajectories")
    else:
        # Initialize MCTS trainer and generate trajectories
        mcts_trainer = MCTSTrainer(game_count=args.mcts_games)
        
        # Generate self-play data
        print(f"Generating {args.mcts_games} self-play games...")
        trajectories = mcts_trainer.train(progress_bar=True)
        print(f"Generated {len(trajectories)} trajectories.")
        
        # Save trajectories if requested
        if args.save_trajectories:
            save_trajectories(trajectories, trajectories_path)
    
    # Quick trajectory analysis
    if trajectories:
        total_samples = sum(len(traj) for game_traj in trajectories for traj in game_traj)
        print(f"Training with {total_samples} total samples from {len(trajectories)} games.")
    
    # Initialize network trainer
    network_trainer = NetworkTrainer(
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        wandb_project=args.wandb_project,
        wandb_api_key=args.wandb_api_key,
        value_update_freq=getattr(args, 'value_update_freq', 2),
        policy_update_freq=getattr(args, 'policy_update_freq', 1)
    )
    
    # Train the network
    print(f"Training network for {args.epochs} epochs...")
    metrics = network_trainer.train(
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_wandb=args.log_wandb
    )
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, "mcts_model.pt")
    network_trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save the training metrics
    if args.plot_losses:
        metrics_plot_path = os.path.join(args.output_dir, "training_losses.png")
        plot_type = 'batch' if getattr(args, 'plot_batch_losses', False) else 'epoch'
        network_trainer.plot_training_metrics(metrics, save_path=metrics_plot_path, plot_type=plot_type)
    
    # Return the network for further use if needed
    return network_trainer, metrics, trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCTS Network")
    parser.add_argument("--mcts-games", type=int, default=100, help="Number of self-play games to generate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--policy-lr", type=float, default=0.001, help="Learning rate for policy network")
    parser.add_argument("--value-lr", type=float, default=0.0001, help="Learning rate for value network")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--plot-losses", action="store_true", help="Plot and save training metrics")
    parser.add_argument("--wandb-project", type=str, default="mcts-training", help="Weights & Biases project name")
    parser.add_argument("--log-wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="log_wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-api-key", type=str, help="Weights & Biases API key")
    parser.add_argument("--value-update-freq", type=int, default=1, help="How often to update value network")
    parser.add_argument("--policy-update-freq", type=int, default=1, help="How often to update policy network")
    parser.add_argument("--trajectories-file", type=str, help="Path to existing trajectories file")
    parser.add_argument("--load-trajectories", action="store_true", help="Load trajectories from file")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectories to file")
    parser.add_argument("--plot-batch-losses", action="store_true", help="Plot batch-level losses instead of epoch-level")
    
    # example usage:
    # Basic training:
    # python train.py --mcts-games 100 --epochs 10 --batch-size 64 --policy-lr 0.001 --value-lr 0.0001 --output-dir "models" --plot-losses --wandb-project "my-project"
    
    # Generate and save trajectories, then train:
    # python train.py --mcts-games 1000 --save-trajectories --epochs 15 --output-dir "models/saved_traj"
    
    # Load existing trajectories and train:
    # python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --epochs 20 --policy-lr 0.01 --value-lr 0.001 --output-dir "models/hp_exp1"
    
    # Train with different update frequencies and batch-level plotting:
    # python train.py --load-trajectories --value-update-freq 1 --policy-update-freq 2 --plot-batch-losses --output-dir "models/update_exp"
    
    # Disable wandb logging:
    # python train.py --no-wandb
    
    args = parser.parse_args()
    main(args)

"""
=== EXAMPLE RUN COMMANDS ===

# Basic Training Examples:
# ------------------------

# 1. Simple training with 100 games, basic settings
python train.py --mcts-games 100 --epochs 10 --output-dir "models/basic"

# 2. Training with trajectory saving for later reuse
python train.py --mcts-games 1000 --save-trajectories --epochs 15 --output-dir "models/saved_traj"

# 3. Higher quality training with more games and epochs
python train.py --mcts-games 2000 --epochs 50 --batch-size 128 --output-dir "models/high_quality"


# Loading and Reusing Trajectories:
# ----------------------------------

# 4. Load existing trajectories and train with different hyperparameters
python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --epochs 20 --policy-lr 0.01 --value-lr 0.001 --output-dir "models/hp_exp1"

# 5. Same trajectories, different update frequencies
python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --value-update-freq 1 --policy-update-freq 2 --output-dir "models/update_exp"

# 6. Load trajectories and train with larger batch size
python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --batch-size 256 --epochs 30 --output-dir "models/large_batch"


# Advanced Training Options:
# ---------------------------

# 7. Training with batch-level loss plotting for detailed analysis
python train.py --mcts-games 500 --plot-losses --plot-batch-losses --output-dir "models/detailed_analysis"

# 8. Load trajectories and plot batch losses with custom update frequencies
python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --plot-losses --plot-batch-losses --value-update-freq 3 --policy-update-freq 1 --output-dir "models/custom_updates"

# 9. Quick experimentation without W&B logging
python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --no-wandb --epochs 5 --output-dir "models/quick_test"


# Weights & Biases Integration:
# ------------------------------

# 10. Training with custom W&B project
python train.py --mcts-games 800 --wandb-project "mangala-mcts-v2" --epochs 25 --output-dir "models/wandb_exp"

# 11. Training with W&B API key (for server environments)
python train.py --mcts-games 1000 --wandb-project "production-mcts" --wandb-api-key "your_api_key_here" --epochs 40 --output-dir "models/production"

# 12. Disable W&B for local testing
python train.py --mcts-games 200 --no-wandb --epochs 10 --output-dir "models/local_test"


# Research and Experimentation Workflows:
# ----------------------------------------

# 13. Generate large dataset once for multiple experiments
python train.py --mcts-games 5000 --save-trajectories --epochs 1 --output-dir "research/dataset_generation"

# 14. Experiment 1: Conservative updates (value network updates more often)
python train.py --load-trajectories --trajectories-file "research/dataset_generation/trajectories_5000games.pkl" --value-update-freq 1 --policy-update-freq 3 --epochs 30 --output-dir "research/conservative_updates"

# 15. Experiment 2: Aggressive policy updates
python train.py --load-trajectories --trajectories-file "research/dataset_generation/trajectories_5000games.pkl" --value-update-freq 3 --policy-update-freq 1 --epochs 30 --output-dir "research/aggressive_policy"

# 16. Experiment 3: Different learning rates
python train.py --load-trajectories --trajectories-file "research/dataset_generation/trajectories_5000games.pkl" --policy-lr 0.01 --value-lr 0.0001 --epochs 30 --output-dir "research/high_policy_lr"

# 17. Final model with best settings and detailed monitoring
python train.py --load-trajectories --trajectories-file "research/dataset_generation/trajectories_5000games.pkl" --value-update-freq 2 --policy-update-freq 1 --policy-lr 0.001 --value-lr 0.0001 --batch-size 64 --epochs 100 --plot-losses --plot-batch-losses --wandb-project "final-mcts-model" --output-dir "models/final"


# Performance and Debugging:
# ---------------------------

# 18. Small scale debugging run
python train.py --mcts-games 10 --epochs 2 --batch-size 4 --no-wandb --output-dir "debug/small_test"

# 19. Memory-efficient training with smaller batches
python train.py --load-trajectories --trajectories-file "models/saved_traj/trajectories_1000games_20241215_143022.pkl" --batch-size 16 --epochs 50 --output-dir "models/memory_efficient"

# 20. GPU training with optimized settings (if CUDA available)
python train.py --mcts-games 2000 --batch-size 512 --epochs 30 --policy-lr 0.01 --value-lr 0.001 --output-dir "models/gpu_optimized"


# Production Training Pipeline:
# ------------------------------

# Step 1: Generate production dataset
python train.py --mcts-games 10000 --save-trajectories --epochs 1 --output-dir "production/data_generation"

# Step 2: Train production model with monitoring
python train.py --load-trajectories --trajectories-file "production/data_generation/trajectories_10000games.pkl" --epochs 200 --batch-size 128 --wandb-project "production-mangala-mcts" --plot-losses --plot-batch-losses --output-dir "production/models"

# Step 3: Fine-tune with different settings
python train.py --load-trajectories --trajectories-file "production/data_generation/trajectories_10000games.pkl" --policy-lr 0.0001 --value-lr 0.00001 --epochs 50 --output-dir "production/fine_tuned"
"""
