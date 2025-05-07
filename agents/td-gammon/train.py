from agents.random_agent import RandomAgent
from mangala.mangala import Mangala
from . import TDTrainer

if __name__ == '__main__':
    agent0 = RandomAgent(53)
    agent1 = RandomAgent(34)

    game = Mangala(
        agent0=agent0,
        agent1=agent1,
    )

    trainer = TDTrainer(
        game=game,
        agent=agent0,
        network=None,
        learning_rate=0.01,
        discount_factor=0.9,
        trace_decay=0.7,
    )

    trainer.train(episodes=15)
    trainer.save_model(filepath="model.pth")

