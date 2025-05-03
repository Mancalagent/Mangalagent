from agents.base_agent import BaseAgent
from utils.util import Util


class Mangala:

    def __init__(self, agent0: BaseAgent, agent1: BaseAgent):
        # agents initialization
        self.agent0 = agent0
        self.agent1 = agent1
        
        # board initialization 
        self.board = [4] * 14
        self.board[6] = 0  # Player 0's store
        self.board[13] = 0  # Player 1's store
        
        # player turn and game over flag
        self.player_turn = 0
        self.game_over = False
        self.extra_turn = False

    def swap_player(self):
        # if extra turn, skip swap
        if self.extra_turn:
            return
        self.player_turn = 1 - self.player_turn

    def make_move(self, pit_index):
        # Translate player's 0-5 choice to actual board position
        if self.player_turn == 1:
            # For player 1, translate pit_index 0-5 to board positions 7-12
            actual_pit_index = pit_index + 7
        else:
            # For player 0, pit_index already corresponds to board positions 0-5
            actual_pit_index = pit_index
        
        # Check if pit index is valid (0-5)
        if pit_index < 0 or pit_index > 5:
            print(f"Invalid pit index: {pit_index}. Please choose 0-5.")
            return False
            
        rocks = self.board[actual_pit_index]
        if rocks == 0:
            print("Cannot move from an empty pit. Please choose another.")
            return False
        
        self.extra_turn = False

        # New logic for single stone
        if rocks == 1:
            next_index = (actual_pit_index + 1) % 14
            player_store = Util.get_player_store(self.player_turn)
            player_pits = Util.get_players_pits(self.player_turn)
            opponent_pits = Util.get_players_pits(1 - self.player_turn)
            
            # Skip opponent's store
            if next_index == (2 - self.player_turn) * 7 - 1:
                next_index = (next_index + 1) % 14
                
            # Move the stone to next pit or store
            self.board[actual_pit_index] = 0
            
            # Check if landing in an empty pit on player's side
            if next_index in player_pits and self.board[next_index] == 0:
                # Check if opposite pit has stones
                opposite_index = 12 - next_index
                if self.board[opposite_index] > 0:  # Only capture if there are stones
                    print(f"Capture! Taking stones from pit {opposite_index % 7}")
                    self.board[player_store] += self.board[opposite_index] + 1  # Add captured stones + last stone
                    self.board[opposite_index] = 0
                    # Leave the landing pit empty
                    return True
                else:
                    # If opposite pit is empty, just place the stone normally
                    self.board[next_index] += 1
            
            # Check if landing in opponent's pit and making it even
            elif next_index in opponent_pits:
                new_count = self.board[next_index] + 1
                if new_count % 2 == 0:  # If even
                    print(f"Even capture! Taking {new_count} stones from opponent's pit {next_index % 7}")
                    self.board[player_store] += new_count
                    self.board[next_index] = 0
                    return True
                else:
                    # Regular move into opponent's pit
                    self.board[next_index] += 1
            
            # Regular move for other cases
            else:
                self.board[next_index] += 1
            
            # If landed in store, get extra turn
            if next_index == player_store:
                self.extra_turn = True
                print("Extra turn! Last stone landed in your store.")
            return True

        # New logic for 2 or more stones
        # Leave one stone in current pit
        self.board[actual_pit_index] = 1
        rocks -= 1
        
        index = actual_pit_index
        while rocks > 0:
            index = (index + 1) % 14
            if index == (2 - self.player_turn) * 7 - 1:
                # Skip opponent's store
                continue
            
            # If this is the last stone
            if rocks == 1:
                player_store = Util.get_player_store(self.player_turn)
                player_pits = Util.get_players_pits(self.player_turn)
                opponent_pits = Util.get_players_pits(1 - self.player_turn)
                
                # Check if landing in an empty pit on player's side
                if index in player_pits and self.board[index] == 0:
                    # Check if opposite pit has stones
                    opposite_index = 12 - index
                    if self.board[opposite_index] > 0:  # Only capture if there are stones
                        print(f"Capture! Taking stones from pit {opposite_index % 7}")
                        self.board[player_store] += self.board[opposite_index] + 1  # Add captured stones + last stone
                        self.board[opposite_index] = 0
                        # Don't add the stone to this pit - it's already counted in player_store
                        rocks -= 1
                        continue
                    else:
                        # If opposite pit is empty, just place the stone normally
                        self.board[index] += 1
                        rocks -= 1
                        continue
                
                # Check if landing in opponent's pit and making it even
                elif index in opponent_pits:
                    new_count = self.board[index] + 1
                    if new_count % 2 == 0:  # If even
                        print(f"Even capture! Taking {new_count} stones from opponent's pit {index % 7}")
                        self.board[player_store] += new_count
                        self.board[index] = 0
                        rocks -= 1
                        continue
                
                # If landing in player's store, get extra turn
                if index == player_store:
                    self.extra_turn = True
                    print("Extra turn! Last stone landed in your store.")

            self.board[index] += 1
            rocks -= 1
        
        return True

    def display_board(self):
        """Display board from current player's perspective"""
        if self.player_turn == 1:
            print("                 Player 1     ")
            print("--------------------------------------------")
            print("       5'    4'    3'    2'    1'    0'")
            print("    +-----+-----+-----+-----+-----+-----+")
            print(f"    | {self.board[5]:2d}  | {self.board[4]:2d}  | {self.board[3]:2d}  | {self.board[2]:2d}  | {self.board[1]:2d}  | {self.board[0]:2d}  |")
            print(f" {self.board[6]:2d} +-----+-----+-----+-----+-----+-----+ {self.board[13]:2d}")
            print(f"    | {self.board[7]:2d}  | {self.board[8]:2d}  | {self.board[9]:2d}  | {self.board[10]:2d}  | {self.board[11]:2d}  | {self.board[12]:2d}  |")
            print("    +-----+-----+-----+-----+-----+-----+")
            print("       0     1     2     3     4     5")
            print("--------------------------------------------")
        else:  
            print("                  Player 0      ")
            print("--------------------------------------------")
            print("       5'    4'    3'    2'    1'    0'")
            print("    +-----+-----+-----+-----+-----+-----+")
            print(f"    | {self.board[12]:2d}  | {self.board[11]:2d}  | {self.board[10]:2d}  | {self.board[9]:2d}  | {self.board[8]:2d}  | {self.board[7]:2d}  |")
            print(f" {self.board[13]:2d} +-----+-----+-----+-----+-----+-----+ {self.board[6]:2d}")
            print(f"    | {self.board[0]:2d}  | {self.board[1]:2d}  | {self.board[2]:2d}  | {self.board[3]:2d}  | {self.board[4]:2d}  | {self.board[5]:2d}  |")
            print("    +-----+-----+-----+-----+-----+-----+")
            print("       0     1     2     3     4     5")
            print("--------------------------------------------")
    def check_game_over(self) -> int:
        p1_pits = self.board[0:6]
        p2_pits = self.board[7:13]
        if sum(p1_pits) == 0 or sum(p2_pits) == 0:
            self.game_over = True
            if sum(p1_pits) == 0:
                self.board[6] += sum(p2_pits)
            else:
                self.board[13] += sum(p1_pits)
            return self.get_winner(), self.board[6], self.board[13]
        return -1, self.board[6], self.board[13]

    def get_winner(self) -> int:
        player0_score = self.board[6]
        player1_score = self.board[13]
        if player0_score > player1_score:
            return 0
        return 1

    def reset(self):
        self.board = [4] * 14
        self.board[6] = 0
        self.board[13] = 0
        self.player_turn = 0
        self.game_over = False
        self.extra_turn = False

    def start(self):
        while not self.game_over:
            current_agent = self.agent0 if self.player_turn == 0 else self.agent1
            print(f"Player {self.player_turn}'s turn")
            
            # Show the board from current player's perspective
            self.display_board()
            
            # Keep asking for input until a valid move is made
            valid_move = False
            while not valid_move:
                # Get move as 0-5 from either player
                move = current_agent.act((self.board, self.player_turn))
                
                # Validate move is within 0-5
                if move < 0 or move > 5:
                    print(f"Invalid move {move}, must be 0-5. Try again.")
                    continue
                    
                # Try to make the move
                valid_move = self.make_move(move)
                
            result, player0_score, player1_score = self.check_game_over()
            if result != -1:
                self.game_over = True
                print(f"Game over! Player {result} wins!")
                print(f"Player 0 score: {player0_score}")
                print(f"Player 1 score: {player1_score}")
                break
            
            self.swap_player()