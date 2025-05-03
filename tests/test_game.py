from datetime import datetime

from agents.base_agent import BaseAgent
from mangala.mangala import Mangala

if __name__ == '__main__':
    time = datetime.now().strftime("%H:%M")
    p0 = BaseAgent(time)
    p1 = BaseAgent(time)
    game = Mangala(p0, p1)
    game.start()