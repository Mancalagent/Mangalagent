import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from agents.mcts.mcts_network import MCTS_Policy_Network, MCTS_Value_Network
from agents.mcts.train.mcts_dataset import create_mcts_dataloader

class NetworkTrainer:
    """
    Trainer for the MCTS policy and value networks
    """
    def __init__(self, lr=0.001, l2_reg=0.0001):
        """
        Initialize the network trainer.
        
        Args:
            network: Not used (kept for backwards compatibility)
            lr: Learning rate
            l2_reg: L2 regularization weight
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy_network = MCTS_Policy_Network(input_dim=14, output_dim=6)
        self.value_network = MCTS_Value_Network(input_dim=14, output_dim=1)
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, weight_decay=l2_reg)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr, weight_decay=l2_reg)
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        
    def reinforce_loss(self, policy_logits, actions, advantages):
        """
        Compute REINFORCE loss (policy gradient)
        
        Args:
            policy_logits: Raw policy network outputs
            actions: One-hot encoded actions that were taken
            advantages: Advantage values (can be rewards or TD errors)
            
        Returns:
            Policy loss using REINFORCE algorithm
        """
        # Convert policy logits to log probabilities
        log_probs = F.log_softmax(policy_logits, dim=1)
        
        # Compute selected action log probabilities
        # We multiply one-hot actions by log probs and sum across action dimension
        selected_log_probs = torch.sum(log_probs * actions, dim=1)
        
        # Compute REINFORCE loss (negative because we want to maximize)
        # We multiply by advantages to weight actions based on their value
        policy_loss = -torch.mean(selected_log_probs * advantages)
        
        return policy_loss
    
    def calculate_policy_f1(self, policy_logits, policy_targets):
        """
        Calculate F1 score for policy predictions
        
        Args:
            policy_logits: Raw policy network outputs
            policy_targets: One-hot encoded ground truth actions
            
        Returns:
            F1 score (harmonic mean of precision and recall)
        """
        # Get predicted actions (argmax)
        pred_actions = torch.argmax(policy_logits, dim=1)
        true_actions = torch.argmax(policy_targets, dim=1)
        
        # Calculate precision and recall components
        true_positives = torch.sum((pred_actions == true_actions).float())
        total_predicted = pred_actions.size(0)  # Total number of predictions
        total_actual = true_actions.size(0)  # Total number of actual positives
        
        # Calculate precision and recall
        precision = true_positives / total_predicted if total_predicted > 0 else 0
        recall = true_positives / total_actual if total_actual > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1.item()
    
    def calculate_value_accuracy(self, value_preds, value_targets, threshold=0.2):
        """
        Calculate value accuracy as the percentage of predictions 
        within a threshold of the true values
        
        Args:
            value_preds: Predicted values
            value_targets: Ground truth values
            threshold: Maximum allowed difference for a prediction to be "correct"
            
        Returns:
            Accuracy score
        """
        # Calculate absolute differences
        abs_diff = torch.abs(value_preds - value_targets)
        
        # Count predictions within threshold
        correct_preds = torch.sum(abs_diff <= threshold).float()
        total_preds = value_preds.size(0)
        
        # Calculate accuracy
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        
        return accuracy.item()
        
    def train(self, trajectories, epochs=10, batch_size=64):
        """
        Train the network on MCTS trajectories.
        
        Args:
            trajectories: List of trajectories from MCTS self-play
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        dataloader = create_mcts_dataloader(trajectories, batch_size=batch_size)
        
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'policy_f1': [],
            'value_accuracy': []
        }
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc="Training Epochs", unit="epoch")
        
        for epoch in epoch_pbar:
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            epoch_policy_f1 = 0
            epoch_value_accuracy = 0
            batches = 0
            
            # Create batch progress bar
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                             leave=False, unit="batch")
            
            for batch in batch_pbar:
                # Move data to device
                states = batch['state'].to(self.device)
                policy_targets = batch['policy_target'].to(self.device)
                value_targets = batch['value_target'].float().to(self.device)
                
                # Forward pass - separate for policy and value networks
                policy_logits = self.policy_network(states)
                value_preds = self.value_network(states).squeeze(-1)
                
                # Calculate losses
                # For policy, use REINFORCE loss (value_targets as advantage)
                policy_loss = self.reinforce_loss(
                    policy_logits=policy_logits,
                    actions=policy_targets,
                    advantages=value_targets  # Using game outcomes as advantages
                )
                
                # For value, use MSE loss
                value_loss = self.value_loss_fn(value_preds, value_targets)
                
                # Combined loss for tracking (not used for optimization)
                total_loss = policy_loss + value_loss
                
                # Calculate accuracy metrics
                policy_f1 = self.calculate_policy_f1(policy_logits, policy_targets)
                value_accuracy = self.calculate_value_accuracy(value_preds, value_targets)
                
                # Backpropagation - separate for each network
                # Policy network update
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # Value network update
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                # Track metrics
                current_policy_loss = policy_loss.item()
                current_value_loss = value_loss.item()
                current_total_loss = current_policy_loss + current_value_loss
                
                epoch_policy_loss += current_policy_loss
                epoch_value_loss += current_value_loss
                epoch_total_loss += current_total_loss
                epoch_policy_f1 += policy_f1
                epoch_value_accuracy += value_accuracy
                batches += 1
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'policy_loss': f"{current_policy_loss:.4f}",
                    'value_loss': f"{current_value_loss:.4f}",
                    'policy_f1': f"{policy_f1:.4f}",
                    'value_acc': f"{value_accuracy:.4f}"
                })
            
            # Average metrics for the epoch
            avg_policy_loss = epoch_policy_loss / batches
            avg_value_loss = epoch_value_loss / batches 
            avg_total_loss = epoch_total_loss / batches
            avg_policy_f1 = epoch_policy_f1 / batches
            avg_value_accuracy = epoch_value_accuracy / batches
            
            metrics['policy_loss'].append(avg_policy_loss)
            metrics['value_loss'].append(avg_value_loss)
            metrics['total_loss'].append(avg_total_loss)
            metrics['policy_f1'].append(avg_policy_f1)
            metrics['value_accuracy'].append(avg_value_accuracy)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'policy_loss': f"{avg_policy_loss:.4f}",
                'value_loss': f"{avg_value_loss:.4f}",
                'policy_f1': f"{avg_policy_f1:.4f}",
                'value_acc': f"{avg_value_accuracy:.4f}"
            })
        
        return metrics
    
    def plot_training_metrics(self, metrics, save_path=None):
        """
        Plot training metrics over epochs.
        
        Args:
            metrics: Dictionary containing lists of loss values
            save_path: Path to save the plot image (optional)
        """
        epochs = range(1, len(metrics['policy_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot losses in one subplot
        plt.subplot(2, 2, 1)
        plt.plot(epochs, metrics['policy_loss'], 'b-', label='Policy Loss')
        plt.plot(epochs, metrics['value_loss'], 'r-', label='Value Loss')
        plt.plot(epochs, metrics['total_loss'], 'g-', label='Total Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy metrics in another subplot
        plt.subplot(2, 2, 2)
        plt.plot(epochs, metrics['policy_f1'], 'b-', label='Policy F1 Score')
        plt.plot(epochs, metrics['value_accuracy'], 'r-', label='Value Accuracy')
        plt.title('Accuracy Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot policy metrics separately
        plt.subplot(2, 2, 3)
        plt.plot(epochs, metrics['policy_loss'], 'b-', label='Loss')
        plt.plot(epochs, metrics['policy_f1'], 'g-', label='F1 Score')
        plt.title('Policy Network')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot value metrics separately
        plt.subplot(2, 2, 4)
        plt.plot(epochs, metrics['value_loss'], 'r-', label='Loss')
        plt.plot(epochs, metrics['value_accuracy'], 'g-', label='Accuracy')
        plt.title('Value Network')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save models to disk"""
        policy_path = path.replace('.pt', '_policy.pt')
        value_path = path.replace('.pt', '_value.pt')
        
        torch.save(self.policy_network.state_dict(), policy_path)
        torch.save(self.value_network.state_dict(), value_path)
        print(f"Policy model saved to {policy_path}")
        print(f"Value model saved to {value_path}")
    
    def load_model(self, path):
        """Load models from disk"""
        policy_path = path.replace('.pt', '_policy.pt')
        value_path = path.replace('.pt', '_value.pt')
        
        self.policy_network.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.value_network.load_state_dict(torch.load(value_path, map_location=self.device))
        
        self.policy_network.to(self.device)
        self.value_network.to(self.device)










    