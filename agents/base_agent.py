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
            input_range = range(p_index * 6, (p_index + 1) * 6)
            pit_index = int(input(f"For player {p_index}, choose a pit index from {input_range}: "))
            while pit_index not in input_range:
                pit_index = int(input(f"Invalid choice. Choose a pit index from {input_range}: "))
            return pit_index
        raise NotImplementedError("This method should be overridden by subclasses.")

    def print_state(self, state):
        board, p_index = state
        print("\nCurrent board state:")

        # Player 1's side (reversed order, pits 12 to 7)
        print("   ", end="")
        for i in range(12, 6, -1):
            print(f"[{board[i]:2}]", end=" ")
        print(f"\n[{board[13]:2}]                           [{board[6]:2}]")  # Mangalas

        # Player 0's side (pits 0 to 5)
        print("   ", end="")
        for i in range(0, 6):
            print(f"[{board[i]:2}]", end=" ")
        print("\n       Player 0 side (bottom)")
        print("       Player 1 side (top)\n")

    def __str__(self):
        return f""
