from agents.base_agent import BaseAgent
from util.util import Util


class Mangala:

    def __init__(self, agent0: BaseAgent, agent1: BaseAgent):
        self.agent0 = agent0
        self.agent1 = agent1
        self.board = [4] * 14
        self.board[6] = 0  # Player 0's store
        self.board[13] = 0  # Player 1's store
        self.player_turn = 0
        self.game_over = False
        self.extra_turn = False

    def swap_player(self):
        if self.extra_turn:
            return
        self.player_turn = 1 - self.player_turn

    def make_move(self, pit_index):
        rocks = self.board[pit_index]
        if rocks == 0:
            return

        self.board[pit_index] = 0
        index = pit_index
        self.extra_turn = False
        while rocks > 0:
            index = (index + 1) % 14
            if index == (2 - self.player_turn) * 7 - 1:
                # Skip opponent's store
                continue
            if rocks == 1:
                player_store = Util.get_player_store(self.player_turn)

                if self.board[index] == 0 and index in Util.get_players_pits(self.player_turn):
                    # Capture opponent's stones
                    opposite_index = 12 - index
                    self.board[player_store] += self.board[opposite_index] + 1
                    self.board[opposite_index] = 0
                    self.board[index] = -1

                if index == player_store:
                    # Extra turn when the last stone lands in your mangala
                    self.extra_turn = True

            self.board[index] += 1
            rocks -= 1

    def check_game_over(self) -> int:
        p1_pits = self.board[0:6]
        p2_pits = self.board[7:13]
        if sum(p1_pits) == 0 or sum(p2_pits) == 0:
            self.game_over = True
            if sum(p1_pits) == 0:
                self.board[6] += sum(p2_pits)
            else:
                self.board[13] += sum(p1_pits)
            return self.get_winner()
        return -1

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
            move = current_agent.act((self.board, self.player_turn))
            self.make_move(move)
            result = self.check_game_over()
            if result != -1:
                self.game_over = True
                print(f"Game over! Player {result} wins!")
                break
            self.swap_player()