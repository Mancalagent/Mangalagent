import enum

from util.util import Util


class AgentState(enum.Enum):
    AUTO = "auto"
    HUMAN = "human"


class BaseAgent:
    def __init__(self, id):
        self.id = id
        self.state = AgentState.HUMAN

    def flip_board(self, state):
        print("Flipping board")
        board, p_index = state
        if p_index == 0:
            return board, p_index

        p1_board = board[0:7]
        p2_board = board[7:14]
        flipped_board = p2_board + p1_board
        return flipped_board, p_index

    def act(self, state) -> int:
        board, p_index = self.flip_board(state)
        Util.save_board_state(self.id,board, p_index)
        if self.state == AgentState.HUMAN:
            self.readable_state(board)
            input_range = Util.get_players_pits(p_index)
            pit_index = int(input(f"For player {p_index}, choose a pit index from {input_range}: ")) + (7 * p_index)
            while pit_index not in input_range:
                pit_index = int(input(f"Invalid choice. Choose a pit index from {input_range}: ")) + (7 * p_index)
            return pit_index
        raise NotImplementedError("This method should be overridden by subclasses.")

    def readable_state(self, board):
        print("\nCurrent board state:")

        # Agents side
        print("   ", end="")
        for i in range(12, 6, -1):
            print(f"[{board[i]:2}] ", end="")
        print()

        # Agents pit indices
        print("   ", end="")
        for i in range(12, 6, -1):
            print(f" {i:2}  ", end="")
        print()

        # Mangalas and spacing
        print(f"[{board[13]:2}]                           [{board[6]:2}]")
        print(f" 13                              6")

        # Opponents side
        print("   ", end="")
        for i in range(0, 6):
            print(f"[{board[i]:2}] ", end="")
        print()

        # Opponents pit indices
        print("   ", end="")
        for i in range(0, 6):
            print(f" {i:2}  ", end="")
        print()


def __str__(self):
    return f""