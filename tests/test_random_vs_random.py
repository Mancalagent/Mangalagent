import sys
import os
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.random_agent import RandomAgent
from mangala.mangala import Mangala

def test_random_vs_random(num_games=100, debug=False):
    num_games = max(2, num_games + num_games % 2)
    
    print(f"\n{'='*50}")
    print(f"STARTING RANDOM VS RANDOM TEST: {num_games} games")
    print(f"{'='*50}")
    
    results = {
        'total': {
            'player0_wins': 0,
            'player1_wins': 0,
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
    
    agent0 = RandomAgent(id="RandomAgent0")
    agent1 = RandomAgent(id="RandomAgent1")
    
    games_per_condition = num_games // 2
    
    print(f"\n{'*'*30}")
    print(f"PHASE 1: Agent0 as FIRST player ({games_per_condition} games)")
    print(f"{'*'*30}")
    
    print("\nRunning Phase 1...")
    for game_num in range(1, games_per_condition + 1):
        if debug:
            print(f"\n{'='*30}")
            print(f"GAME {game_num}/{games_per_condition} (Agent0 first)")
            print(f"{'='*30}")
        
        game = Mangala(agent0, agent1, debug=debug)
        
        try:
            start_time = time.time()
            winner = game.start(max_moves=200)
            end_time = time.time()
            
            game_time = end_time - start_time
            move_count = game.move_count if hasattr(game, 'move_count') else 'unknown'
            
            if winner == 2:
                if debug:
                    print("Game ended in a draw!")
                results['total']['draws'] += 1
                results['as_first']['draws'] += 1
            else:
                if winner == 0:
                    if debug:
                        print("Agent0 wins!")
                    results['total']['player0_wins'] += 1
                    results['as_first']['wins'] += 1
                else:
                    if debug:
                        print("Agent1 wins!")
                    results['total']['player1_wins'] += 1
                    results['as_first']['losses'] += 1
            
            if debug:
                print(f"Game completed in {game_time:.2f} seconds")
            results['total']['moves'].append(move_count)
            results['as_first']['moves'].append(move_count)
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'*'*30}")
    print(f"PHASE 2: Agent1 as FIRST player ({games_per_condition} games)")
    print(f"{'*'*30}")
    
    print("\nRunning Phase 2...")
    for game_num in range(1, games_per_condition + 1):
        if debug:
            print(f"\n{'='*30}")
            print(f"GAME {game_num}/{games_per_condition} (Agent1 first)")
            print(f"{'='*30}")
        
        game = Mangala(agent1, agent0, debug=debug)
        
        try:
            start_time = time.time()
            winner = game.start(max_moves=200)
            end_time = time.time()
            
            game_time = end_time - start_time
            move_count = game.move_count if hasattr(game, 'move_count') else 'unknown'
            
            if winner == 2:
                if debug:
                    print("Game ended in a draw!")
                results['total']['draws'] += 1
                results['as_second']['draws'] += 1
            else:
                if winner == 1:
                    if debug:
                        print("Agent1 wins!")
                    results['total']['player1_wins'] += 1
                    results['as_second']['wins'] += 1
                else:
                    if debug:
                        print("Agent0 wins!")
                    results['total']['player0_wins'] += 1
                    results['as_second']['losses'] += 1
            
            if debug:
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
    print(f"Agent0 wins: {results['total']['player0_wins']}")
    print(f"Agent1 wins: {results['total']['player1_wins']}")
    print(f"Draws: {results['total']['draws']}")
    
    if results['total']['moves'] and any(isinstance(m, (int, float)) for m in results['total']['moves']):
        valid_moves = [m for m in results['total']['moves'] if isinstance(m, (int, float))]
        if valid_moves:
            avg_moves = sum(valid_moves) / len(valid_moves)
            print(f"Average moves per game: {avg_moves:.1f}")
    
    print(f"\nAgent0 win rate: {results['total']['player0_wins'] / num_games * 100:.1f}%")
    print(f"Agent1 win rate: {results['total']['player1_wins'] / num_games * 100:.1f}%")
    
    print("\nAGENT0 AS FIRST PLAYER:")
    print("-" * 20)
    total_games = len(results['as_first']['moves'])
    if total_games > 0:
        print(f"Wins: {results['as_first']['wins']}/{total_games} "
              f"({results['as_first']['wins'] / total_games * 100:.1f}%)")
        print(f"Losses: {results['as_first']['losses']}/{total_games} "
              f"({results['as_first']['losses'] / total_games * 100:.1f}%)")
        print(f"Draws: {results['as_first']['draws']}/{total_games} "
              f"({results['as_first']['draws'] / total_games * 100:.1f}%)")
        
        valid_moves = [m for m in results['as_first']['moves'] if isinstance(m, (int, float))]
        if valid_moves:
            avg_moves = sum(valid_moves) / len(valid_moves)
            print(f"Average moves: {avg_moves:.1f}")
    
    print("\nAGENT1 AS FIRST PLAYER:")
    print("-" * 20)
    total_games = len(results['as_second']['moves'])
    if total_games > 0:
        print(f"Wins: {results['as_second']['wins']}/{total_games} "
              f"({results['as_second']['wins'] / total_games * 100:.1f}%)")
        print(f"Losses: {results['as_second']['losses']}/{total_games} "
              f"({results['as_second']['losses'] / total_games * 100:.1f}%)")
        print(f"Draws: {results['as_second']['draws']}/{total_games} "
              f"({results['as_second']['draws'] / total_games * 100:.1f}%)")
        
        valid_moves = [m for m in results['as_second']['moves'] if isinstance(m, (int, float))]
        if valid_moves:
            avg_moves = sum(valid_moves) / len(valid_moves)
            print(f"Average moves: {avg_moves:.1f}")
    
    print("\n" + "="*50)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    test_random_vs_random(num_games=args.games, debug=args.debug)
