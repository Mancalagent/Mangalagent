import os
import sys

from agents.human_agent import HumanAgent


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangala.mangala import Mangala

if __name__ == '__main__':
    p0 = HumanAgent("player0")
    p1 = HumanAgent("player1")
    game = Mangala(p0, p1)
    game.start()
