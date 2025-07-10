import random
import numpy as np
from agents.base_agent import BaseAgent
from mangala.mangala import Mangala
from utils.util import Util

class MinimaxAgent(BaseAgent):
    """Planning agent using minimax algorithm for Mangala"""
    def __init__(self, id, max_depth=5, verbose=False):
        super().__init__(id)
        self.max_depth = max_depth
        self.verbose = verbose
        
    def act(self, state):
        board, player_turn = state
        available_actions = self.get_available_actions(board)
        
        # Safety check - always return a valid move
        if not available_actions:
            return 0  # Default to first pit as fallback
        if len(available_actions) == 1:
            return available_actions[0]
        
        # Core minimax search
        best_action = available_actions[0]  # Default to first available action
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        
        # Try each action and evaluate
        for action in available_actions:
            board_copy = board.copy()
            extra_turn = Mangala.check_for_extra_turn(board_copy, action)
            new_board, reward, is_terminal = Mangala.transition(board_copy, action)
            
            # Calculate value of this move
            if extra_turn and not is_terminal:
                # If extra turn, remain as maximizer
                value = reward*10 + self.minimax(new_board, 0, alpha, beta, True)
            else:
                # If no extra turn, opponent's perspective (minimizer)
                value = reward*10 + self.minimax(Mangala.flip_board(new_board), 0, alpha, beta, False)
            
            if self.verbose:
                print(f"Action {action} has value {value}")
                
            # Update best action
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
        
        # Safety check - ensure we never return None
        if best_action is None and available_actions:
            best_action = available_actions[0]
            
        return best_action
        
    def minimax(self, board, depth, alpha, beta, is_maximizing):
        # Terminal conditions
        if Mangala.check_terminal(board) or depth >= self.max_depth:
            return self.evaluate(board)
            
        available_actions = self.get_available_actions(board)
        if not available_actions:
            return self.evaluate(board)
        
        # Maximizing player (current player)
        if is_maximizing:
            value = float("-inf")
            
            # Check for moves that give extra turns first
            for action in available_actions:
                # Extra turn moves are considered first for better pruning
                if Mangala.check_for_extra_turn(board, action):
                    new_board, reward, is_terminal = Mangala.transition(board.copy(), action)
                    if not is_terminal:
                        child_value = reward*10 + self.minimax(new_board, depth+1, alpha, beta, True)
                        value = max(value, child_value)
                        alpha = max(alpha, value)
                        if beta <= alpha:
                            break
            
            # Regular moves
            for action in available_actions:
                if not Mangala.check_for_extra_turn(board, action):
                    new_board, reward, is_terminal = Mangala.transition(board.copy(), action)
                    child_value = reward*10 + self.minimax(Mangala.flip_board(new_board), depth+1, alpha, beta, False)
                    value = max(value, child_value)
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                        
            return value
        # Minimizing player (opponent)
        else:
            value = float("inf")
            
            # Check for moves that give extra turns first
            for action in available_actions:
                # Extra turn moves for opponent are considered first
                if Mangala.check_for_extra_turn(board, action):
                    new_board, reward, is_terminal = Mangala.transition(board.copy(), action)
                    if not is_terminal:
                        child_value = reward*10 + self.minimax(new_board, depth+1, alpha, beta, False)
                        value = min(value, child_value)
                        beta = min(beta, value)
                        if beta <= alpha:
                            break
            
            # Regular moves
            for action in available_actions:
                if not Mangala.check_for_extra_turn(board, action):
                    new_board, reward, is_terminal = Mangala.transition(board.copy(), action)
                    child_value = reward*10 + self.minimax(Mangala.flip_board(new_board), depth+1, alpha, beta, True)
                    value = min(value, child_value)
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                        
            return value
            
    def evaluate(self, board):
        # Basic components of evaluation
        player_store = 6
        opponent_store = 13
        
        # Terminal state - actual score difference
        if Mangala.check_terminal(board):
            score_diff = board[player_store] - board[opponent_store]
            return score_diff * 100  # High weight for terminal states
            
        # Current score difference (most important factor)
        score_diff = board[player_store] - board[opponent_store]
        
        # Count stones on each side
        player_stones = sum(board[0:6])
        opponent_stones = sum(board[7:13])
        
        # Mangala-specific positional advantages
        position_value = 0
        
        # 1. Extra turn potential - stones that can land in store
        for i in range(6):
            if board[i] == (6 - i):  # Exact landing in store
                position_value += 8
                
        # 2. Capture potential - empty pits with stones opposite
        for i in range(6):
            if board[i] == 0:  # Empty pit
                opposite = 12 - i
                if board[opposite] > 0:  # Has stones opposite
                    position_value += 3 * board[opposite]  # Value proportional to stones
                    
        # 3. Keep stones in pits close to our store
        for i in range(6):
            position_value += board[i] * (0.5 * (5 - i))  # More weight to pits close to store
            
        # 4. Prefer to have more stones on our side than opponent's
        stone_balance = player_stones - opponent_stones
        
        # Combine all factors with appropriate weights for final evaluation
        evaluation = (
            score_diff * 10 +    # Current score difference (high weight)
            position_value +     # Positional advantages
            stone_balance * 2    # Stone balance
        )
        
        return evaluation
