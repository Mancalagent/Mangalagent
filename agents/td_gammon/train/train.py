from datetime import datetime

from agents.base_agent import BaseAgent
from agents.td_gammon.td_trainer import TDTrainer

if __name__ == '__main__':
    agent0 = BaseAgent(53)
    trainer = TDTrainer(
        agent=agent0,
        network=None,
        learning_rate=1e-3,
        discount_factor=1,
        trace_decay=0.7,
    )

    trainer.train(episodes=1000)
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_model(filepath=f"{name}_model.pth")

