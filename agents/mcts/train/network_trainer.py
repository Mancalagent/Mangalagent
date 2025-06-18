import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
from agents.mcts.mcts_network import MCTS_Policy_Network, MCTS_Value_Network
from agents.mcts.train.mcts_dataset import create_mcts_dataloader

class NetworkTrainer:
    """
    Trainer for the MCTS policy and value networks
    """
    def __init__(self, policy_lr=0.001, value_lr=0.0001, wandb_project="mcts-training", wandb_api_key=None, 
                 value_update_freq=2, policy_update_freq=1):
        """
        Initialize the network trainer.
        
        Args:
            policy_lr: Learning rate for policy network
            value_lr: Learning rate for value network
            wandb_project: Weights & Biases project name
            wandb_api_key: Weights & Biases API key (optional, can also use wandb login or env var)
            value_update_freq: How often to update value network (e.g., 2 means every 2 batches)
            policy_update_freq: How often to update policy network (e.g., 1 means every batch)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy_network = MCTS_Policy_Network(input_dim=14, output_dim=6)
        self.value_network = MCTS_Value_Network(input_dim=14, output_dim=1)
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_lr)
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        
        # Update frequencies
        self.value_update_freq = value_update_freq
        self.policy_update_freq = policy_update_freq
        
        # Initialize wandb
        self.wandb_project = wandb_project
        self.wandb_api_key = wandb_api_key
        
    def reinforce_loss(self, policy_logits, actions, advantages):
       
        
        log_probs = F.log_softmax(policy_logits, dim=1)  


        selected_log_probs = torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(1)  
        
       
        if advantages.numel() > 1: 
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages_normalized = advantages
        
    
        policy_loss = -torch.mean(selected_log_probs * advantages_normalized.detach())
        
        return policy_loss
        
    def train(self, trajectories, epochs=10, batch_size=64, log_wandb=True):
        """
        Train the network on MCTS trajectories.
        
        Args:
            trajectories: List of trajectories from MCTS self-play
            epochs: Number of training epochs
            batch_size: Batch size for training
            log_wandb: Whether to log metrics to Weights & Biases
            
        Returns:
            Dictionary with training metrics
        """
        # Initialize wandb if logging is enabled
        if log_wandb:
            # Login with API key if provided
            if self.wandb_api_key:
                wandb.login(key=self.wandb_api_key)
            
            wandb.init(
                project=self.wandb_project,
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "policy_lr": self.policy_optimizer.param_groups[0]['lr'],
                    "value_lr": self.value_optimizer.param_groups[0]['lr'],
                    "value_update_freq": self.value_update_freq,
                    "policy_update_freq": self.policy_update_freq,
                    "device": str(self.device)
                }
            )
        
        dataloader = create_mcts_dataloader(trajectories, batch_size=batch_size)
        
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            # Add batch-level metrics
            'batch_policy_loss': [],
            'batch_value_loss': [],
            'batch_total_loss': [],
            'batch_epochs': []  # Track which epoch each batch belongs to
        }
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc="Training Epochs", unit="epoch")
        
        # Track batch numbers for update frequency
        global_batch_count = 0
        
        for epoch in epoch_pbar:
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            batches = 0
            policy_updates = 0
            value_updates = 0
            
            # Create batch progress bar
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                             leave=False, unit="batch")
            
            for batch in batch_pbar:
                global_batch_count += 1
                
                # Move data to device
                states = batch['state'].to(self.device)
                policy_targets = batch['policy_target'].to(self.device)
                value_targets = batch['value_target'].float().to(self.device)
                
                # Always compute forward passes for both networks
                policy_logits = self.policy_network(states) # outputs logits for each action
                value_preds = self.value_network(states) # outputs value between -1 and 1
                
                # Compute losses
                value_loss = self.value_loss_fn(value_targets, value_preds.squeeze()) # MSE loss
                advantage = (value_targets - value_preds.squeeze()).detach()  # TD error as advantage
                
                policy_loss = self.reinforce_loss(
                    policy_logits=policy_logits,
                    actions=policy_targets,
                    advantages=advantage
                )
                
                # Determine whether to update each network based on frequency
                update_value = (global_batch_count % self.value_update_freq == 0)
                update_policy = (global_batch_count % self.policy_update_freq == 0)
                
                # VALUE NETWORK UPDATE
                if update_value:
                    self.value_optimizer.zero_grad()
                    value_loss.backward(retain_graph=True)  # retain graph for potential policy update
                    self.value_optimizer.step()
                    value_updates += 1
                
                # POLICY NETWORK UPDATE
                if update_policy:
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()
                    policy_updates += 1
                
                # Track metrics (always record losses even if not updating)
                current_policy_loss = policy_loss.item()
                current_value_loss = value_loss.item()
                current_total_loss = current_policy_loss + current_value_loss
                
                # Store batch-level metrics
                metrics['batch_policy_loss'].append(current_policy_loss)
                metrics['batch_value_loss'].append(current_value_loss)
                metrics['batch_total_loss'].append(current_total_loss)
                metrics['batch_epochs'].append(epoch + 1)
                
                epoch_policy_loss += current_policy_loss
                epoch_value_loss += current_value_loss
                epoch_total_loss += current_total_loss
                batches += 1
                
                # Log batch-level metrics to wandb
                if log_wandb:
                    wandb.log({
                        "batch": len(metrics['batch_policy_loss']),
                        "epoch": epoch + 1,
                        "batch_policy_loss": current_policy_loss,
                        "batch_value_loss": current_value_loss,
                        "batch_total_loss": current_total_loss,
                        "value_updated": update_value,
                        "policy_updated": update_policy
                    })
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'policy_loss': f"{current_policy_loss:.4f}",
                    'value_loss': f"{current_value_loss:.4f}",
                    'total_loss': f"{current_total_loss:.4f}",
                    'V_updates': value_updates,
                    'P_updates': policy_updates
                })
            
            # Average metrics for the epoch
            avg_policy_loss = epoch_policy_loss / batches
            avg_value_loss = epoch_value_loss / batches 
            avg_total_loss = epoch_total_loss / batches
            
            metrics['policy_loss'].append(avg_policy_loss)
            metrics['value_loss'].append(avg_value_loss)
            metrics['total_loss'].append(avg_total_loss)
            
            # Log epoch-level metrics to wandb
            if log_wandb:
                wandb.log({
                    "epoch_policy_loss": avg_policy_loss,
                    "epoch_value_loss": avg_value_loss,
                    "epoch_total_loss": avg_total_loss
                })
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'policy_loss': f"{avg_policy_loss:.4f}",
                'value_loss': f"{avg_value_loss:.4f}",
                'total_loss': f"{avg_total_loss:.4f}"
            })
        
        if log_wandb:
            wandb.finish()
        
        return metrics
    
    def plot_training_metrics(self, metrics, save_path=None, plot_type='epoch'):
        """
        Plot training metrics over epochs or batches.
        
        Args:
            metrics: Dictionary containing lists of loss values
            save_path: Path to save the plot image (optional)
            plot_type: 'epoch' for epoch-level metrics, 'batch' for batch-level metrics
        """
        if plot_type == 'batch':
            self._plot_batch_metrics(metrics, save_path)
        else:
            self._plot_epoch_metrics(metrics, save_path)
    
    def _plot_epoch_metrics(self, metrics, save_path=None):
        """Plot epoch-level training metrics."""
        epochs = range(1, len(metrics['policy_loss']) + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(epochs, metrics['policy_loss'], 'b-', label='Policy Loss')
        plt.plot(epochs, metrics['value_loss'], 'r-', label='Value Loss')
        plt.plot(epochs, metrics['total_loss'], 'g-', label='Total Loss')
        plt.title('Training Losses (Epoch-level)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot policy loss separately
        plt.subplot(2, 2, 2)
        plt.plot(epochs, metrics['policy_loss'], 'b-', label='Policy Loss')
        plt.title('Policy Network Loss (Epoch-level)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot value loss separately
        plt.subplot(2, 2, 3)
        plt.plot(epochs, metrics['value_loss'], 'r-', label='Value Loss')
        plt.title('Value Network Loss (Epoch-level)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot total loss separately
        plt.subplot(2, 2, 4)
        plt.plot(epochs, metrics['total_loss'], 'g-', label='Total Loss')
        plt.title('Total Loss (Epoch-level)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Epoch-level training metrics plot saved to {save_path}")
        
        plt.show()
    
    def _plot_batch_metrics(self, metrics, save_path=None):
        """Plot batch-level training metrics."""
        if 'batch_policy_loss' not in metrics or len(metrics['batch_policy_loss']) == 0:
            print("No batch-level metrics found. Make sure to train with batch tracking enabled.")
            return
        
        batch_numbers = range(1, len(metrics['batch_policy_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot all losses together
        plt.subplot(2, 2, 1)
        plt.plot(batch_numbers, metrics['batch_policy_loss'], 'b-', alpha=0.7, label='Policy Loss')
        plt.plot(batch_numbers, metrics['batch_value_loss'], 'r-', alpha=0.7, label='Value Loss')
        plt.plot(batch_numbers, metrics['batch_total_loss'], 'g-', alpha=0.7, label='Total Loss')
        plt.title('Training Losses (Batch-level)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add epoch boundaries
        if 'batch_epochs' in metrics:
            epoch_boundaries = []
            current_epoch = metrics['batch_epochs'][0]
            for i, epoch in enumerate(metrics['batch_epochs']):
                if epoch != current_epoch:
                    epoch_boundaries.append(i)
                    current_epoch = epoch
            
            for boundary in epoch_boundaries:
                plt.axvline(x=boundary, color='black', linestyle=':', alpha=0.5)
        
        # Plot policy loss separately
        plt.subplot(2, 2, 2)
        plt.plot(batch_numbers, metrics['batch_policy_loss'], 'b-', alpha=0.8, label='Policy Loss')
        plt.title('Policy Network Loss (Batch-level)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add epoch boundaries
        for boundary in epoch_boundaries if 'batch_epochs' in metrics else []:
            plt.axvline(x=boundary, color='black', linestyle=':', alpha=0.5)
        
        # Plot value loss separately
        plt.subplot(2, 2, 3)
        plt.plot(batch_numbers, metrics['batch_value_loss'], 'r-', alpha=0.8, label='Value Loss')
        plt.title('Value Network Loss (Batch-level)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add epoch boundaries
        for boundary in epoch_boundaries if 'batch_epochs' in metrics else []:
            plt.axvline(x=boundary, color='black', linestyle=':', alpha=0.5)
        
        # Plot total loss separately
        plt.subplot(2, 2, 4)
        plt.plot(batch_numbers, metrics['batch_total_loss'], 'g-', alpha=0.8, label='Total Loss')
        plt.title('Total Loss (Batch-level)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add epoch boundaries
        for boundary in epoch_boundaries if 'batch_epochs' in metrics else []:
            plt.axvline(x=boundary, color='black', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Batch-level training metrics plot saved to {save_path}")
        
        plt.show()
        
    def plot_batch_training_metrics(self, metrics, save_path=None):
        """
        Convenience method to plot batch-level training metrics.
        
        Args:
            metrics: Dictionary containing lists of loss values
            save_path: Path to save the plot image (optional)
        """
        self.plot_training_metrics(metrics, save_path, plot_type='batch')
    
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










    