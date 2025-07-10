import torch

from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from agents.td_gammon import TDNetwork
from mangala.mangala import Mangala
from agents.mcts_agent import MCTSAgent
from utils.util import Util

weight_path = "/training/models/replay.pth"
def td_gammon_vs_random():

    network = TDNetwork()
    network.load_state_dict(torch.load(weight_path))
    agent0 = TDAgent(53, network)
    agent1 = RandomAgent(34)
    game_count = 1000
    win = 0
    for i in range(game_count):
        #print(f"Game {i+1}/{game_count}")
        game = Mangala(
            agent0=agent0,
            agent1=agent1,
        )
        game.start()
        if game.get_winner() == 0:
            win += 1
    print(f"Win rate: {(win / game_count)*100 :.2f}%")

def td_gammon_vs_random_second():
    network = TDNetwork()
    network.load_state_dict(torch.load(weight_path))
    agent0 = RandomAgent(34)
    agent1 = TDAgent(53, network)
    game_count = 1000
    win = 0
    for i in range(game_count):
        #print(f"Game {i+1}/{game_count}")
        game = Mangala(
            agent0=agent0,
            agent1=agent1,
        )
        game.start()
        if game.get_winner() == 1:
            win += 1
    print(f"Win rate: {(win / game_count)*100 :.2f}%")

def td_gammon_vs_human():
    network = TDNetwork()
    network.load_state_dict(torch.load(weight_path))
    agent0 = TDAgent(53, network)
    agent1 = HumanAgent(34)
    game = Mangala(
        agent0=agent0,
        agent1=agent1,
        debug=True
    )
    game.start()
    
def mcts_vs_random():
    agent0 = MCTSAgent(53, model_path="/kuacc/users/mgoksu21/COMP438/Mangalagent/50k_models/mcts_model_policy.pt")
    agent1 = RandomAgent(34)
    game = Mangala(
        agent0=agent0,
        agent1=agent1,
    )
    game.start()
    
    
def human_vs_human():
    agent0 = HumanAgent(53)
    agent1 = HumanAgent(34)
    game = Mangala(
        agent0=agent0,
        agent1=agent1,
        
    )
    
    game.start()

def td_w_mcst_vs_random():
    tree = Util.load_tree()
    agent1 = MCTSAgent(53, mcts_tree=tree)

def state_test():
    network = TDNetwork()
    network.load_state_dict(torch.load(weight_path))
    agent0 = TDAgent(53, network)
    agent1 = RandomAgent(34)

    state = [0, 0, 0, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 0, 4]
    game = Mangala(
        agent0=agent0,
        agent1=agent1,
        board=state,
        debug= True
    )
    game.start()



if __name__ == '__main__':
    td_gammon_vs_random()
    td_gammon_vs_random_second()
    #td_gammon_vs_human()
    # mcts_vs_random()
    #human_vs_human()
    # pass
    #state_test()