import enum

from utils.util import Util


class AgentState(enum.Enum):
    AUTO = "auto"
    HUMAN = "human"


class BaseAgent:
    def __init__(self, id):
        self.id = id
        self.state = AgentState.HUMAN

    def flip_board(self, state):
        board, p_index = state
        if p_index == 0:
            return board, p_index

        p1_board = board[0:7]
        p2_board = board[7:14]
        flipped_board = p2_board + p1_board
        return flipped_board, p_index

    def act(self, state) -> int:
        board, p_index = self.flip_board(state)
        Util.save_board_state(self.id, board, p_index)
        if self.state == AgentState.HUMAN:
            # Always ask for input 0-5 regardless of player
            pit_index = int(input(f"Player {p_index}, choose a pit (0-5): "))
            while pit_index < 0 or pit_index > 5:
                pit_index = int(input(f"Invalid choice. Choose a pit from 0-5: "))
            return pit_index  # Return 0-5, Mangala class will handle translation
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __str__(self):
        return f"Agent({self.id})"