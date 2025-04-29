import enum


class AgentState(enum.Enum):
    AUTO = "auto"
    HUMAN = "human"


class BaseAgent:
    def __init__(self):
        self.state = AgentState.HUMAN

    def act(self, state):
        board, p_index = state
        if self.state == AgentState.HUMAN:
            self.readable_state(board)
            input_range = range((p_index * 6) + 1, ((p_index + 1) * 6) + 1)
            pit_index = int(input(f"For player {p_index}, choose a pit index from {input_range}: "))
            while pit_index not in input_range:
                pit_index = int(input(f"Invalid choice. Choose a pit index from {input_range}: "))
            return pit_index
        raise NotImplementedError("This method should be overridden by subclasses.")

    def readable_state(self, board):
        print("\nCurrent board state:")

        # Player 1's side
        print("   ", end="")
        for i in range(12, 6, -1):
            print(f"[{board[i]:2}] ", end="")
        print()

        # Player 1's pit indices
        print("   ", end="")
        for i in range(12, 6, -1):
            print(f" {i:2}  ", end="")
        print()

        # Mangalas and spacing
        print(f"[{board[13]:2}]                           [{board[6]:2}]")
        print(f" 13                              6")

        # Player 0's side
        print("   ", end="")
        for i in range(0, 6):
            print(f"[{board[i]:2}] ", end="")
        print()

        # Player 0's pit indices
        print("   ", end="")
        for i in range(0, 6):
            print(f" {i:2}  ", end="")
        print()

        print("\n       Player 0 side (bottom)")
        print("       Player 1 side (top)\n")


def __str__(self):
    return f""
