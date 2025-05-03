import sys
import os

# Add the parent directory to sys.path to find modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from mangala.mangala import Mangala

if __name__ == '__main__':
    p0 = BaseAgent("player0")
    p1 = BaseAgent("player1")
    game = Mangala(p0, p1)
    game.start()
