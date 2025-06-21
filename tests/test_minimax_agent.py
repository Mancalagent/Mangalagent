import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from mangala.mangala import Mangala

def test_minimax_vs_random(num_games=5, debug=False):
    """
    Test the MinimaxAgent against a RandomAgent
    
    Args:
        num_games: Number of games to play
        debug: Whether to print the board state during games or not
    """
    print(f"Playing {num_games} games: MinimaxAgent vs RandomAgent")
    
    # Track wins
    minimax_wins = 0
    random_wins = 0
    draws = 0
    
    # Create agents
    minimax_agent = MinimaxAgent(id="MinimaxAgent", max_depth=3)
    random_agent = RandomAgent(id="RandomAgent")
    
    for game in range(num_games):
        print(f"\nGame {game+1}/{num_games}")
        
        # Alternate who goes first
        if game % 2 == 0:
            game = Mangala(minimax_agent, random_agent, debug=debug)
            print("MinimaxAgent is Player 0")
        else:
            game = Mangala(random_agent, minimax_agent, debug=debug)
            print("MinimaxAgent is Player 1")
            
        start_time = time.time()
        game.start()
        end_time = time.time()
        
        winner = game.get_winner()
        if winner == 2:
            print(f"Game ended in a draw!")
            draws += 1
        else:
            agent_name = "MinimaxAgent" if (
                (game.agent0.id == "MinimaxAgent" and winner == 0) or
                (game.agent1.id == "MinimaxAgent" and winner == 1)
            ) else "RandomAgent"
            
            print(f"Winner: {agent_name}")
            
            if agent_name == "MinimaxAgent":
                minimax_wins += 1
            else:
                random_wins += 1
                
        print(f"Final scores - Player 0: {game.board[6]}, Player 1: {game.board[13]}")
        print(f"Game duration: {end_time - start_time:.2f} seconds")

    print("\n===== RESULTS =====")
    print(f"MinimaxAgent wins: {minimax_wins}")
    print(f"RandomAgent wins: {random_wins}")
    print(f"Draws: {draws}")
    print(f"MinimaxAgent win rate: {minimax_wins/num_games:.2f}")

def play_against_minimax():
    """
    Play against the MinimaxAgent as the user
    """
    print("Play against MinimaxAgent!")
    print("You are Player 0, MinimaxAgent is Player 1")
    
    human_agent = HumanAgent(id="Human")
    minimax_agent = MinimaxAgent(id="MinimaxAgent", max_depth=3)
    game = Mangala(human_agent, minimax_agent, debug=True)
    game.start()
    winner = game.get_winner()
    if winner == 2:
        print("Game ended in a draw!")
    elif winner == 0:
        print("You win!")
    else:
        print("MinimaxAgent wins!")
        
    print(f"Final scores - You: {game.board[6]}, MinimaxAgent: {game.board[13]}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the MinimaxAgent")
    parser.add_argument("--mode", type=str, choices=["auto", "play"], default="auto",
                        help="'auto' to watch games between MinimaxAgent and RandomAgent, 'play' to play against MinimaxAgent")
    parser.add_argument("--games", type=int, default=5,
                        help="Number of games to play in auto mode")
    parser.add_argument("--debug", action="store_true",
                        help="Print board state during games")
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        test_minimax_vs_random(num_games=args.games, debug=args.debug)
    else:
        play_against_minimax()
