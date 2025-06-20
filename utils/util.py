import base64
import json
import pickle
from pathlib import Path

from agents.mcts.mcts_tree import MCTSTree


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

    @classmethod
    def save_tree(cls, tree, file_path='mcts_tree.json'):
        """
        Save the MCTS tree to a JSON file using pickle serialization.

        Args:
            file_path: Path to save the JSON file.
        """
        try:
            # Serialize the MCTS tree using pickle
            serialized_tree = pickle.dumps(tree)

            # Convert the serialized data to a JSON-compatible format (base64 encoding)
            encoded_tree = base64.b64encode(serialized_tree).decode('utf-8')

            # Save the encoded tree to a JSON file
            with open(file_path, 'w') as json_file:
                json.dump({'mcts_tree': encoded_tree}, json_file)

            print(f"MCTS tree saved to {file_path}")
        except Exception as e:
            print(f"Error saving MCTS tree: {e}")

    @classmethod
    def load_tree(cls, file_path='mcts_tree.json') -> MCTSTree | None:
        """
        Load the MCTS tree from a JSON file.

        Args:
            file_path: Path to the JSON file containing the MCTS tree.

        Returns:
            An instance of MCTSTree.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                encoded_tree = data['mcts_tree']

                # Decode the base64 encoded tree
                serialized_tree = base64.b64decode(encoded_tree)

                # Deserialize the MCTS tree using pickle
                tree = pickle.loads(serialized_tree)

                print(f"MCTS tree loaded from {file_path}")
                return tree
        except Exception as e:
            print(f"Error loading MCTS tree: {e}")
            return None
