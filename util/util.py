from datetime import datetime
from pathlib import Path


class Util:
    @classmethod
    def get_players_pits(self, player_turn):
        if player_turn == 0:
            return range(0, 6)
        return range(7, 13)

    @classmethod
    def get_player_store(cls, player_turn):
        if player_turn == 0:
            return 6
        return 13

    @classmethod
    def save_board_state(cls, name, board, p_index):
        game_history_dir = Path("./game_history")
        game_history_dir.mkdir(parents=True, exist_ok=True)

        with open(game_history_dir / f"{name}.txt", "a") as f:
            f.write(f"State: {(board, p_index)}\n")
            f.write(f"-" * 20 + "\n")