from datetime import datetime

from agents.base_agent import BaseAgent
from agents.td_gammon.td_trainer import TDTrainer

if __name__ == '__main__':
    agent0 = BaseAgent(53)
    trainer = TDTrainer(
        agent=agent0,
        network=None,
        learning_rate=1e-3,
        discount_factor=0.9,
        trace_decay=0.7,
    )

    trainer.train(episodes=100_000)
    trainer.save_model(filepath=f"model.pth")
