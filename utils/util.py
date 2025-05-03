from datetime import datetime
from pathlib import Path


class Util:
    @classmethod
    def get_players_pits(cls, player_turn):
        """
        Get the pit indices for the given player.
        Note: This still returns the actual board indices (0-5 or 7-12)
        not the player-facing indices (always 0-5)
        """
        if player_turn == 0:
            return range(0, 6)
        return range(7, 13)

    @classmethod
    def get_player_store(cls, player_turn):
        """Get the store index for the given player"""
        if player_turn == 0:
            return 6
        return 13

    @classmethod
    def save_board_state(cls, name, board, p_index):
        """Save the current board state to a file for debugging/analysis"""
        game_history_dir = Path("./tests/game_history")
        game_history_dir.mkdir(parents=True, exist_ok=True)

        with open(game_history_dir / f"{name}.txt", "a") as f:
            f.write(f"State: {(board, p_index)}\n")
            f.write(f"-" * 20 + "\n")