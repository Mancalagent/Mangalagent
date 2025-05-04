import enum

from utils.util import Util


# class AgentState(enum.Enum):
#     AUTO = "auto"
#     HUMAN = "human"


class BaseAgent:
    def __init__(self, id):
        self.id = id
        # self.agent_state = AgentState.HUMAN
        self.game_state = None
        self.player_index = None

    def act(self, state) -> int:
        return NotImplementedError("This method should be overridden by subclasses.")

    def get_current_state(self):
        return self.game_state

    def get_available_actions(self, state):
        board, _ = state
        return [i for i in range(6) if board[i] > 0]

    def __str__(self):
        return f"Agent({self.id})"
