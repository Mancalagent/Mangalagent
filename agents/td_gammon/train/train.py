from datetime import datetime

import os
from agents.base_agent import BaseAgent
from agents.td_gammon.td_trainer import TDTrainer

if __name__ == '__main__':
    # Create directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    agent0 = BaseAgent(53)
    trainer = TDTrainer(
        agent=agent0,
        network=None,
        learning_rate=1e-3,
        discount_factor=0.9,
        trace_decay=0.7,
    )

    # Train the model
    print("Starting training...")
    trainer.train(episodes=1000)
    
    # Save the trained model
    model_path = 'models/td_gammon_model.pth'
    trainer.save_model(filepath=model_path)
    print(f"\nTraining complete! Model saved to {model_path}")
