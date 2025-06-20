import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from mangala.mangala import Mangala

def test_minimax_vs_random(num_games=5, debug=False, max_depth=3):
    # Ensure even number of games for balanced first/second player distribution
    num_games = max(2, num_games + num_games % 2)
    
    print(f"\n{'='*50}")
    print(f"STARTING TEST: {num_games} games, Minimax depth={max_depth}")
    print(f"{'='*50}")
    
    # Track overall results
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
    
    # Create agents
    minimax_agent = MinimaxAgent(id="MinimaxAgent", max_depth=max_depth, verbose=debug)
    random_agent = RandomAgent(id="RandomAgent")
    
    # First half: Minimax as first player, second half: Minimax as second player
    games_per_condition = num_games // 2
    
    print(f"\n{'*'*30}")
    print(f"PHASE 1: Minimax as FIRST player ({games_per_condition} games)")
    print(f"{'*'*30}")
    
    # Test with Minimax as first player
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
            
            # Process results
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
            
            # Store game stats
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
    
    # Test with Minimax as second player
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
            
            # Process results
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
            
            # Store game stats
            print(f"Game completed in {game_time:.2f} seconds")
            results['total']['moves'].append(move_count)
            results['as_second']['moves'].append(move_count)
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
            
    # Helper function to calculate average moves
    def avg_moves(moves_list):
        valid = [m for m in moves_list if isinstance(m, int)]
        return sum(valid)/len(valid) if valid else 0
    
    # Print detailed results
    print("\n" + "="*50)
    print("TEST COMPLETED")
    print("="*50)
    
    # Overall results
    print("\nOVERALL RESULTS:")
    print("-"*20)
    print(f"Total games: {num_games}")
    print(f"Minimax wins: {results['total']['minimax_wins']}")
    print(f"Random wins: {results['total']['random_wins']}")
    print(f"Draws: {results['total']['draws']}")
    
    if results['total']['moves']:
        avg = avg_moves(results['total']['moves'])
        print(f"Average moves per game: {avg:.1f}")
    
    win_rate = results['total']['minimax_wins'] / num_games * 100
    print(f"\nOverall Minimax win rate: {win_rate:.1f}%")
    
    # Results when Minimax is first
    print("\nMINIMAX AS FIRST PLAYER:")
    print("-"*20)
    games_as_first = games_per_condition
    wins_as_first = results['as_first']['wins']
    losses_as_first = results['as_first']['losses']
    draws_as_first = results['as_first']['draws']
    
    print(f"Wins: {wins_as_first}/{games_as_first} ({wins_as_first/games_as_first*100:.1f}%)")
    print(f"Losses: {losses_as_first}/{games_as_first} ({losses_as_first/games_as_first*100:.1f}%)")
    print(f"Draws: {draws_as_first}/{games_as_first} ({draws_as_first/games_as_first*100:.1f}%)")
    if results['as_first']['moves']:
        avg = avg_moves(results['as_first']['moves'])
        print(f"Average moves: {avg:.1f}")
    
    # Results when Minimax is second
    print("\nMINIMAX AS SECOND PLAYER:")
    print("-"*20)
    games_as_second = games_per_condition
    wins_as_second = results['as_second']['wins']
    losses_as_second = results['as_second']['losses']
    draws_as_second = results['as_second']['draws']
    
    print(f"Wins: {wins_as_second}/{games_as_second} ({wins_as_second/games_as_second*100:.1f}%)")
    print(f"Losses: {losses_as_second}/{games_as_second} ({losses_as_second/games_as_second*100:.1f}%)")
    print(f"Draws: {draws_as_second}/{games_as_second} ({draws_as_second/games_second*100:.1f}%)")
    if results['as_second']['moves']:
        avg = avg_moves(results['as_second']['moves'])
        print(f"Average moves: {avg:.1f}")
    
    print("="*50)
    
    return results

def play_against_minimax():
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
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to play in auto mode")
    parser.add_argument("--debug", action="store_true",
                        help="Print board state during games")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum depth for minimax search")
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        test_minimax_vs_random(
            num_games=args.games, 
            debug=args.debug, 
            max_depth=args.max_depth
        )
    else:
        play_against_minimax()
