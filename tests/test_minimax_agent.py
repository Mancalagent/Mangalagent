import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from mangala.mangala import Mangala

def test_minimax_vs_random(num_games=100, debug=False, max_depth=3):
    num_games = max(2, num_games + num_games % 2)
    
    print(f"\n{'='*50}")
    print(f"STARTING TEST: {num_games} games, Minimax depth={max_depth}")
    print(f"{'='*50}")
    
    results = {
        'total': {
            'minimax_wins': 0,
            'random_wins': 0,
            'draws': 0,
            'moves': []
        },
        'as_first': {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'moves': []
        },
        'as_second': {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'moves': []
        }
    }
    
    minimax_agent = MinimaxAgent(id="MinimaxAgent", max_depth=max_depth, verbose=debug)
    random_agent = RandomAgent(id="RandomAgent")
    
    games_per_condition = num_games // 2
    
    print(f"\n{'*'*30}")
    print(f"PHASE 1: Minimax as FIRST player ({games_per_condition} games)")
    print(f"{'*'*30}")
    
    for game_num in range(1, games_per_condition + 1):
        print(f"\n{'='*30}")
        print(f"GAME {game_num}/{games_per_condition} (Minimax first)")
        print(f"{'='*30}")
        
        game = Mangala(minimax_agent, random_agent, debug=debug)
        minimax_player = 0
        
        try:
            start_time = time.time()
            winner = game.start(max_moves=200)
            end_time = time.time()
            
            game_time = end_time - start_time
            move_count = game.move_count if hasattr(game, 'move_count') else 'unknown'
            
            if winner == 2:
                print("Game ended in a draw!")
                results['total']['draws'] += 1
                results['as_first']['draws'] += 1
            else:
                if winner == minimax_player:
                    print("MinimaxAgent wins!")
                    results['total']['minimax_wins'] += 1
                    results['as_first']['wins'] += 1
                else:
                    print("RandomAgent wins!")
                    results['total']['random_wins'] += 1
                    results['as_first']['losses'] += 1
            
            print(f"Game completed in {game_time:.2f} seconds")
            results['total']['moves'].append(move_count)
            results['as_first']['moves'].append(move_count)
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'*'*30}")
    print(f"PHASE 2: Minimax as SECOND player ({games_per_condition} games)")
    print(f"{'*'*30}")
    
    for game_num in range(1, games_per_condition + 1):
        print(f"\n{'='*30}")
        print(f"GAME {game_num}/{games_per_condition} (Minimax second)")
        print(f"{'='*30}")
        
        game = Mangala(random_agent, minimax_agent, debug=debug)
        minimax_player = 1
        
        try:
            start_time = time.time()
            winner = game.start(max_moves=200)
            end_time = time.time()
            
            game_time = end_time - start_time
            move_count = game.move_count if hasattr(game, 'move_count') else 'unknown'
            
            if winner == 2:
                print("Game ended in a draw!")
                results['total']['draws'] += 1
                results['as_second']['draws'] += 1
            else:
                if winner == minimax_player:
                    print("MinimaxAgent wins!")
                    results['total']['minimax_wins'] += 1
                    results['as_second']['wins'] += 1
                else:
                    print("RandomAgent wins!")
                    results['total']['random_wins'] += 1
                    results['as_second']['losses'] += 1
            
            print(f"Game completed in {game_time:.2f} seconds")
            results['total']['moves'].append(move_count)
            results['as_second']['moves'].append(move_count)
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("TEST COMPLETED")
    print("="*50 + "\n")
    
    print("OVERALL RESULTS:")
    print("-" * 20)
    print(f"Total games: {num_games}")
    print(f"Minimax wins: {results['total']['minimax_wins']}")
    print(f"Random wins: {results['total']['random_wins']}")
    print(f"Draws: {results['total']['draws']}")
    
    if results['total']['moves']:
        avg_moves = sum(results['total']['moves']) / len(results['total']['moves'])
        print(f"Average moves per game: {avg_moves:.1f}")
    
    print(f"\nOverall Minimax win rate: {results['total']['minimax_wins'] / num_games * 100:.1f}%")
    
    print("\nMINIMAX AS FIRST PLAYER:")
    print("-" * 20)
    total_games = len(results['as_first']['moves'])
    if total_games > 0:
        print(f"Wins: {results['as_first']['wins']}/{total_games} ({results['as_first']['wins'] / total_games * 100:.1f}%)")
        print(f"Losses: {results['as_first']['losses']}/{total_games} ({results['as_first']['losses'] / total_games * 100:.1f}%)")
        print(f"Draws: {results['as_first']['draws']}/{total_games} ({results['as_first']['draws'] / total_games * 100:.1f}%)")
        
        if results['as_first']['moves']:
            avg_moves = sum(results['as_first']['moves']) / len(results['as_first']['moves'])
            print(f"Average moves: {avg_moves:.1f}")
    
    print("\nMINIMAX AS SECOND PLAYER:")
    print("-" * 20)
    total_games = len(results['as_second']['moves'])
    if total_games > 0:
        print(f"Wins: {results['as_second']['wins']}/{total_games} ({results['as_second']['wins'] / total_games * 100:.1f}%)")
        print(f"Losses: {results['as_second']['losses']}/{total_games} ({results['as_second']['losses'] / total_games * 100:.1f}%)")
        print(f"Draws: {results['as_second']['draws']}/{total_games} ({results['as_second']['draws'] / total_games * 100:.1f}%)")
        
        if results['as_second']['moves']:
            avg_moves = sum(results['as_second']['moves']) / len(results['as_second']['moves'])
            print(f"Average moves: {avg_moves:.1f}")
    
    print("\n" + "="*50)
    return results

def play_against_minimax():
    print("\n" + "="*50)
    print("PLAYING AGAINST MINIMAX AGENT")
    print("="*50 + "\n")
    
    depth = int(input("Enter minimax depth (default 3): ") or 3)
    human_first = input("Do you want to go first? (y/n, default y): ").lower() != 'n'
    
    minimax_agent = MinimaxAgent(id="MinimaxAgent", max_depth=depth, verbose=True)
    human_agent = HumanAgent(id="Human")
    
    if human_first:
        game = Mangala(human_agent, minimax_agent, debug=True)
    else:
        game = Mangala(minimax_agent, human_agent, debug=True)
    
    game.start()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "play"], default="auto")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_depth", type=int, default=3)
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        test_minimax_vs_random(num_games=args.games, debug=args.debug, max_depth=args.max_depth)
    else:
        play_against_minimax()
