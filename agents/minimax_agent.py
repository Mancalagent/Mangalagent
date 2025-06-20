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
            
        # Opening book - pit 2 or 3 are good first moves
        if sum(board) == 48 and board[6] == 0 and board[13] == 0:
            # First move of the game
            if 2 in available_actions:
                return 2
            elif 3 in available_actions:
                return 3
        
        # Use iterative deepening with time management for better move selection
        best_action = available_actions[0]  # Default to first available action
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        
        # Order moves for better alpha-beta pruning
        ordered_actions = self.order_moves(board, available_actions, True)
        
        # Try each action and evaluate using minimax
        for action in ordered_actions:
            board_copy = board.copy()
            extra_turn = Mangala.check_for_extra_turn(board_copy, action)
            new_board, reward, is_terminal = Mangala.transition(board_copy, action)
            
            # Calculate value of this move
            if is_terminal:
                # If move leads to terminal state, evaluate directly
                value = self.evaluate(new_board)
            elif extra_turn:
                # If extra turn, remain as maximizer
                value = reward * 10 + self.minimax(new_board, 0, alpha, beta, True)
            else:
                # If no extra turn, opponent's perspective (minimizer)
                value = reward * 10 + self.minimax(
                    Mangala.flip_board(new_board), 0, alpha, beta, False
                )
            
            if self.verbose:
                print(f"Action {action} has value {value}")
                
            # Update best action
            if value > best_value or (value == best_value and random.random() < 0.3):
                best_value = value
                best_action = action
                
            # Update alpha for alpha-beta pruning
            alpha = max(alpha, best_value)
            
            # Early exit if we find a guaranteed win
            if best_value >= 1000:
                break
        
        return best_action
        
    def order_moves(self, board, actions, is_maximizing):
        """Order moves to improve alpha-beta pruning efficiency."""
        if not actions:
            return actions
            
        # Score each move based on heuristics
        scored_moves = []
        for action in actions:
            score = 0
            
            # Extra turns are very good for us, bad for opponent
            if Mangala.check_for_extra_turn(board, action):
                score += 1000 if is_maximizing else -1000
                
            # Captures are good
            new_board = board.copy()
            last_pit = (action + new_board[action]) % 14
            if (0 <= last_pit < 6 and new_board[last_pit] == 0 and 
                new_board[12 - last_pit] > 0):
                score += 50 + new_board[12 - last_pit] * 10
                
            # Moving stones closer to store is generally good
            if action <= 5:  # Only for player's side
                score += (6 - action) * 2
                
            scored_moves.append((score, action))
        
        # Sort by score (descending for maximizer, ascending for minimizer)
        scored_moves.sort(reverse=is_maximizing, key=lambda x: x[0])
        return [move[1] for move in scored_moves]
    
    def minimax(self, board, depth, alpha, beta, is_maximizing):
        # Debug output
        if self.verbose and depth <= 1:  # Only show top few depths to avoid too much output
            print(f"{'  ' * depth}Depth {depth}: {'Max' if is_maximizing else 'Min'}, Alpha: {alpha}, Beta: {beta}")
            
        # Check for terminal state or max depth
        if Mangala.check_terminal(board):
            val = self.evaluate(board)
            if self.verbose and depth <= 1:
                print(f"{'  ' * depth}Terminal state, value: {val}")
            return val
            
        available_actions = self.get_available_actions(board)
        if not available_actions or depth >= self.max_depth:
            val = self.evaluate(board)
            if self.verbose and depth <= 1:
                print(f"{'  ' * depth}Max depth or no actions, value: {val}")
            return val
        
        # Order moves for better alpha-beta pruning
        ordered_actions = self.order_moves(board, available_actions, is_maximizing)
        
        if is_maximizing:
            max_eval = float('-inf')
            
            for action in ordered_actions:
                board_copy = board.copy()
                extra_turn = Mangala.check_for_extra_turn(board_copy, action)
                new_board, reward, is_terminal = Mangala.transition(board_copy, action)
                
                if extra_turn and not is_terminal:
                    # Keep maximizing if extra turn
                    eval = reward * 10 + self.minimax(new_board, depth, alpha, beta, True)
                else:
                    # Switch to minimizing if no extra turn
                    eval = reward * 10 + self.minimax(
                        Mangala.flip_board(new_board), depth + 1, alpha, beta, False
                    )
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            return max_eval
            
        else:  # Minimizing player
            min_eval = float('inf')
            
            for action in ordered_actions:
                board_copy = board.copy()
                extra_turn = Mangala.check_for_extra_turn(board_copy, action)
                new_board, reward, is_terminal = Mangala.transition(board_copy, action)
                
                if extra_turn and not is_terminal:
                    # Keep minimizing if extra turn
                    eval = reward * 10 + self.minimax(new_board, depth, alpha, beta, False)
                else:
                    # Switch to maximizing if no extra turn
                    eval = reward * 10 + self.minimax(
                        Mangala.flip_board(new_board), depth + 1, alpha, beta, True
                    )
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            return min_eval
            
    def evaluate(self, board):
        # Store indices
        player_store = 6
        opponent_store = 13
        
        # Terminal state - actual score difference
        if Mangala.check_terminal(board):
            player_total = board[player_store] + sum(board[0:6])
            opponent_total = board[opponent_store] + sum(board[7:13])
            if player_total > opponent_total:
                return 1000  # Big win
            elif player_total < opponent_total:
                return -1000  # Big loss
            else:
                return 0  # Draw
        
        # 1. Store difference (most important)
        store_diff = board[player_store] - board[opponent_store]
        
        # 2. Stone count difference (material advantage)
        player_stones = sum(board[0:6])
        opponent_stones = sum(board[7:13])
        stone_diff = player_stones - opponent_stones
        
        # 3. Positional advantages
        position_value = 0
        
        # a. Extra turn potential (very valuable)
        for i in range(6):
            if board[i] == (6 - i):  # Exact landing in store
                position_value += 15  # Increased from 8
        
        # b. Capture opportunities
        for i in range(6):
            if board[i] == 0:  # Empty pit
                opposite = 12 - i
                if 0 < board[opposite] <= 5:  # Can capture with the right move
                    position_value += 5 + board[opposite]  # Higher reward for more stones
        
        # c. Potential mobility (empty pits are bad, they reduce options)
        player_moves = sum(1 for i in range(6) if board[i] > 0)
        opponent_moves = sum(1 for i in range(7, 13) if board[i] > 0)
        mobility = player_moves - opponent_moves
        
        # d. Stone distribution (prefer more stones closer to store)
        distribution = 0
        for i in range(6):
            if board[i] > 0:
                # More weight to stones closer to store
                distribution += (6 - i) * board[i]
        
        # 4. Endgame consideration (when few stones left)
        total_stones = player_stones + opponent_stones
        endgame_factor = 1.0
        if total_stones < 15:  # Approaching endgame
            endgame_factor = 2.0  # Increase importance of stone count
        
        # Combine all factors with weights
        evaluation = (
            store_diff * 20 +               # Store difference (most important)
            stone_diff * 3 * endgame_factor +  # Material advantage
            position_value * 2 +            # Positional factors
            mobility * 5 +                   # Mobility advantage
            distribution * 0.5               # Stone distribution
        )
        
        return evaluation
