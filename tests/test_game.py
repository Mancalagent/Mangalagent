from datetime import datetime

from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent

from mangala.mangala import Mangala

if __name__ == '__main__':
    time = datetime.now().strftime("%H:%M")
    p0 = RandomAgent(time)
    p1 = RandomAgent(time)
    game = Mangala(p0, p1)
    game.start()