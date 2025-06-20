from agents.base_agent import BaseAgent
from utils.util import Util


class Mangala:

    def __init__(self, agent0: BaseAgent, agent1: BaseAgent,board=None,debug=False):
        # agents initialization
        self.agent0 = agent0
        self.agent1 = agent1

        if not board:
            self.board = [4] * 14
            self.board[6] = 0  # Player 0's store
            self.board[13] = 0  # Player 1's store
        else:
            self.board = board

        # player turn and game over flag
        self.player_turn = 0
        self.game_over = False
        self.extra_turn = False
        self.debug = debug

    def swap_player(self):
        # if extra turn, skip swap
        if self.extra_turn:
            return
        self.player_turn = 1 - self.player_turn
        if not self.extra_turn:
            self.board = self.flip_board(self.board)

    @classmethod

    def transition(cls, state, action) -> tuple[[int], float, bool]:
        state = state.copy()
        rocks = state[action]
        if rocks == 0:
            raise ValueError("Invalid action: No stones in the selected pit.")

        if rocks == 1:
            state[action] = 0
        else:
            rocks -= 1
            state[action] = 1

        index = action
        player_turn = 0
        player_store = Util.get_player_store(player_turn)
        player_pits = Util.get_players_pits(player_turn)
        opponent_store = Util.get_player_store(1 - player_turn)
        opponent_pits = Util.get_players_pits(1 - player_turn)

        initial_rocks = state[player_store]
        reward = 0
        while rocks > 0:
            index = (index + 1) % 14
            if index == opponent_store:
                continue

            if rocks == 1:
                # Check if landing in an empty pit on player's side
                if index in player_pits and state[index] == 0:
                    opposite_index = 12 - index
                    if state[opposite_index] > 0:  # Only capture if there are stones
                        state[player_store] += state[opposite_index] + 1  # Add captured stones + last stone
                        state[opposite_index] = 0
                        # Don't add the stone to this pit - it's already counted in player_store
                        rocks -= 1
                        reward+= 5
                        continue
                    else:
                        # If opposite pit is empty, just place the stone normally
                        state[index] += 1
                        rocks -= 1
                        continue

                elif index in opponent_pits:
                    #print(f"Landing on opponent's pit {index % 7}")
                    new_count = state[index] + 1
                    if new_count % 2 == 0:
                        #print(f"Even capture! Taking {new_count} stones from opponent's pit {index % 7}")
                        state[player_store] += new_count
                        state[index] = 0
                        rocks -= 1
                        reward+= 5
                        continue

            state[index] += 1
            rocks -= 1
        is_terminal = Mangala.check_game_over(state)
        reward += (state[player_store] - initial_rocks) + (is_terminal==True)*100
        reward = reward / 100
        return state, reward, is_terminal
    
    @classmethod
    def check_terminal(cls, board) -> bool:
        return Mangala.check_game_over(board)
    

    @classmethod
    def check_for_extra_turn(cls,state,pit_index) -> bool:
        rocks = state[pit_index]
        if rocks != 1:
            rocks -= 1

        player_store = 6
        if (rocks + pit_index) == player_store:
            return True
        return False

    def make_move(self, pit_index) -> None:
        self.extra_turn = False
        #print(f"Player {self.player_turn} chooses pit {pit_index}")
        extra_turn = Mangala.check_for_extra_turn(self.board,pit_index)
        if extra_turn:
            self.extra_turn = True
            if self.debug:
                print(f"Player {self.player_turn} gets an extra turn!")

        new_board,_,is_terminal = Mangala.transition(self.board, pit_index)
        if is_terminal:
            self.game_over = True
        self.board = new_board
        self.swap_player()

    def display_board(self):
        print(f"Player {self.player_turn}'s turn")
        """Display board from current player's perspective"""
        print("--------------------------------------------")
        print("       5'    4'    3'    2'    1'    0'")
        print("    +-----+-----+-----+-----+-----+-----+")
        print(
            f"    | {self.board[12]:2d}  | {self.board[11]:2d}  | {self.board[10]:2d}  | {self.board[9]:2d}  | {self.board[8]:2d}  | {self.board[7]:2d}  |")
        print(f" {self.board[13]:2d} +-----+-----+-----+-----+-----+-----+ {self.board[6]:2d}")
        print(
            f"    | {self.board[0]:2d}  | {self.board[1]:2d}  | {self.board[2]:2d}  | {self.board[3]:2d}  | {self.board[4]:2d}  | {self.board[5]:2d}  |")
        print("    +-----+-----+-----+-----+-----+-----+")
        print("       0     1     2     3     4     5")
        print(f"                  Player {self.player_turn}      ")
        print("--------------------------------------------")

    @classmethod
    def check_game_over(self,board) -> bool:
        p1_pits = board[0:6]
        p2_pits = board[7:13]
        game_over = False
        if sum(p1_pits) == 0 or sum(p2_pits) == 0:
            game_over = True
            if sum(p1_pits) == 0:
                board[6] += sum(p2_pits)
            else:
                board[13] += sum(p1_pits)
            board[0:6] = [0] * 6
            board[7:13] = [0] * 6

        return game_over

    def get_winner(self) -> int:
        player0_score = self.board[6]
        player1_score = self.board[13]
        if player0_score > player1_score:
            return 0
        elif player0_score < player1_score:
            return 1
        else:
            return 2

    def reset(self):
        self.board = [4] * 14
        self.board[6] = 0
        self.board[13] = 0
        self.player_turn = 0
        self.game_over = False
        self.extra_turn = False
    @classmethod
    def flip_board(cls,board):
        board = board.copy()
        board_1 = board[0:7]
        board_2 = board[7:14]
        board = board_2 + board_1
        return board


    def start(self, max_moves=200):
        """Start the game with a move limit to prevent infinite games.
        
        Args:
            max_moves: Maximum number of moves before the game is declared a draw
            
        Returns:
            int: 0 if player 0 wins, 1 if player 1 wins, 2 if it's a draw
        """
        move_count = 0
        
        while not self.game_over and move_count < max_moves:
            move_count += 1
            current_agent = self.agent0 if self.player_turn == 0 else self.agent1
            
            if self.debug:
                print(f"\nMove {move_count} - Player {self.player_turn}'s turn")
                self.display_board()
                
            try:
                move = current_agent.act((self.board, self.player_turn))
                if self.debug:
                    print(f"Player {self.player_turn} chooses move: {move}")
                self.make_move(move)
                
                if self.game_over:
                    winner = self.get_winner()
                    if self.debug:
                        print("\n" + "="*50)
                        print(f"Game over after {move_count} moves!")
                        print(f"Player {winner} wins!")
                        print(f"Player 0 score: {self.board[6]}")
                        print(f"Player 1 score: {self.board[13]}")
                        print("="*50)
                    return winner
                    
            except Exception as e:
                print(f"Error during move {move_count}: {str(e)}")
                # In case of error, just make a random move
                import random
                valid_moves = [i for i, stones in enumerate(self.board[:6] if self.player_turn == 0 else self.board[7:13], 
                                                          start=0 if self.player_turn == 0 else 7) if stones > 0]
                if valid_moves:
                    move = random.choice(valid_moves)
                    print(f"Falling back to random move: {move}")
                    self.make_move(move)
        
        # If we get here, the game didn't end naturally
        if not self.game_over:
            print(f"\nGame stopped after reaching maximum of {max_moves} moves")
            # Count stones to determine winner
            p0_score = self.board[6] + sum(self.board[0:6])
            p1_score = self.board[13] + sum(self.board[7:13])
            
            if p0_score > p1_score:
                winner = 0
            elif p1_score > p0_score:
                winner = 1
            else:
                winner = 2  # Draw
                
            if self.debug:
                print("\n" + "="*50)
                print("Game ended by move limit!")
                print(f"Player 0 total: {p0_score}")
                print(f"Player 1 total: {p1_score}")
                print(f"Player {winner} wins!" if winner != 2 else "It's a draw!")
                print("="*50)
                
            return winner
