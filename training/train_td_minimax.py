import os
from agents.base_agent import BaseAgent
from agents.td_gammon.td_trainer_minimax import TDTrainerMinimax

if __name__ == '__main__':
    # Create directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    agent0 = BaseAgent(id=53)
    minimax_depth = 15
    trainer = TDTrainerMinimax(
        agent=agent0,
        network=None,
        learning_rate=1e-3,
        discount_factor=0.9,
        trace_decay=0.7,
        minimax_depth=minimax_depth
    )

    # Train the model
    print(f"Starting training with Minimax (depth={minimax_depth}) exploration...")
    trainer.train(episodes=1000)

    # Save the trained model
    model_path = f'models/td_gammon_minimax_model_depth{minimax_depth}.pth'
    trainer.save_model(filepath=model_path)
    print(f"\nTraining complete! Model saved to {model_path}")
