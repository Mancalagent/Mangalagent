import os
import torch
from agents.random_agent import RandomAgent
from agents.td_agent import TDAgent
from agents.td_gammon.td_network import TDNetwork
from mangala.mangala import Mangala

def test_td_vs_random(num_games=100):
    print(f"Testing TD-Gammon vs Random agent for {num_games} games...")
    
    # Load the trained model
    model_path = "models/td_gammon_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run the training script first.")
    
    network = TDNetwork()
    network.load_state_dict(torch.load(model_path))
    network.eval()  # Set to evaluation mode
    
    # Create agents
    td_agent = TDAgent(agent_id=0, network=network)
    random_agent = RandomAgent(id=1)  # RandomAgent uses id
    
    # Track results
    td_wins = 0
    random_wins = 0
    draws = 0
    
    for game_num in range(1, num_games + 1):
        # Alternate which agent goes first
        if game_num % 2 == 0:
            agent0, agent1 = td_agent, random_agent
        else:
            agent0, agent1 = random_agent, td_agent
        
        # Play the game
        game = Mangala(agent0=agent0, agent1=agent1, debug=False)
        game.start()
        
        # Get the winner
        winner = game.get_winner()
        
        # Update counts
        if winner == -1:
            draws += 1
        elif (winner == 0 and game_num % 2 == 0) or (winner == 1 and game_num % 2 == 1):
            td_wins += 1
        else:
            random_wins += 1
        
        # Print progress
        if game_num % 10 == 0:
            print(f"After {game_num} games: TD-Gammon: {td_wins}, Random: {random_wins}, Draws: {draws}")
    
    # Print final results
    print("\nFinal Results:")
    print(f"TD-Gammon wins: {td_wins} ({td_wins/num_games*100:.1f}%)")
    print(f"Random wins: {random_wins} ({random_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

if __name__ == "__main__":
    test_td_vs_random(num_games=100)
