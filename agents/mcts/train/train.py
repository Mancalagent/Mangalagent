import torch
import os
import argparse
from tqdm.auto import tqdm
from agents.mcts.train.mcts_trainer import MCTSTrainer
from agents.mcts.train.network_trainer import NetworkTrainer

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize MCTS trainer
    mcts_trainer = MCTSTrainer(game_count=args.mcts_games)
    
    # Generate self-play data
    print(f"Generating {args.mcts_games} self-play games...")
    trajectories = mcts_trainer.train(progress_bar=True)
    print(f"Generated {len(trajectories)} trajectories.")
    
    # Initialize network trainer
    network_trainer = NetworkTrainer(lr=args.learning_rate)
    
    # Train the network
    print(f"Training network for {args.epochs} epochs...")
    metrics = network_trainer.train(
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, "mcts_model.pt")
    network_trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save the training metrics
    if args.plot_losses:
        metrics_plot_path = os.path.join(args.output_dir, "training_losses.png")
        network_trainer.plot_training_metrics(metrics, save_path=metrics_plot_path)
    
    # Return the network for further use if needed
    return network_trainer.network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCTS Network")
    parser.add_argument("--mcts-games", type=int, default=100, help="Number of self-play games to generate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--plot-losses", action="store_true", help="Plot and save training metrics")
    
    args = parser.parse_args()
    main(args)
