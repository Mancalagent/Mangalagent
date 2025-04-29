from agents.base_agent import BaseAgent
from mangala.mangala import Mangala

if __name__ == '__main__':
    p0 = BaseAgent()
    p1 = BaseAgent()
    game = Mangala(p0, p1)
    game.start()