import random
import numpy as np
from agents.base_agent import BaseAgent
from mangala.mangala import Mangala
from utils.util import Util

class MinimaxAgent(BaseAgent):
    def __init__(self, id, max_depth=5, verbose=False):
        super().__init__(id)
        self.max_depth = max_depth
        self.verbose = verbose
        self.nodes_evaluated = 0
        self.cache = {}  # For transposition table
        
    def _is_capture_move(self, board, action):
        """Check if a move results in a capture."""
        if board[action] == 0:
            return False
            
        # Simulate the move
        last_pit = (action + board[action]) % 14
        
        # Check if landing in an empty pit on player's side
        if 0 <= last_pit < 6 and board[last_pit] == 0:
            opposite_pit = 12 - last_pit
            if 0 < opposite_pit < 6 and board[opposite_pit] > 0:
                return True
                
        # Check for even capture on opponent's side
        if 7 <= last_pit < 13 and board[last_pit] % 2 == 0:
            return True
            
        return False
        
    def act(self, state):
        board, player_turn = state
        available_actions = self.get_available_actions(board)
        
        if not available_actions:
            return 0  
        if len(available_actions) == 1:
            return available_actions[0]
            
        if sum(board) == 48 and board[6] == 0 and board[13] == 0:
            if 2 in available_actions:
                return 2
            elif 3 in available_actions:
                return 3
        
        best_action = available_actions[0]  
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        
        ordered_actions = self.order_moves(board, available_actions, True)
        
        for action in ordered_actions:
            board_copy = board.copy()
            extra_turn = Mangala.check_for_extra_turn(board_copy, action)
            new_board, reward, is_terminal = Mangala.transition(board_copy, action)
            
            if is_terminal:
                value = self.evaluate(new_board)
            elif extra_turn:
                value = reward * 10 + self.minimax(new_board, 0, alpha, beta, True)
            else:
                value = reward * 10 + self.minimax(
                    Mangala.flip_board(new_board), 0, alpha, beta, False
                )
            
            if self.verbose:
                print(f"Action {action} has value {value}")
                
            if value > best_value or (value == best_value and random.random() < 0.3):
                best_value = value
                best_action = action
                
            alpha = max(alpha, best_value)
            
            if best_value >= 1000:
                break
        
        return best_action
        
    def order_moves(self, board, actions, is_maximizing):
        if not actions:
            return actions
            
        scored_moves = []
        player_store = 6 if is_maximizing else 13
        
        for action in actions:
            score = 0
            
            # 1. Extra turns are extremely valuable
            if Mangala.check_for_extra_turn(board, action):
                score += 5000 if is_maximizing else -5000
            
            # 2. Check for immediate wins
            new_board = board.copy()
            last_pit = (action + new_board[action]) % 14
            
            # 3. Capture opportunities
            if (0 <= last_pit < 6 and new_board[last_pit] == 0 and 
                new_board[12 - last_pit] > 0):
                # The more stones we can capture, the better
                captured = new_board[12 - last_pit]
                score += 100 + captured * 15
            
            # 4. Positional advantages
            if action <= 5:  # Only for player's side
                # Prefer moves that move stones toward the store
                score += (6 - action) * 3
                
                # Prefer moves that empty pits with few stones
                if 0 < board[action] < 3:
                    score += 5 - board[action]
            
            # 5. Deny opponent opportunities
            opponent_store = 13 if is_maximizing else 6
            if board[action] > 0:
                # Calculate if this move could give opponent a good response
                potential_landing = (action + board[action]) % 14
                if potential_landing in range(7, 13):
                    if board[potential_landing] == 0:  # Opponent could capture
                        score -= 30
            
            scored_moves.append((score, action))
        
        # Sort by score (descending for maximizer, ascending for minimizer)
        scored_moves.sort(reverse=is_maximizing, key=lambda x: x[0])
        
        # Return only the top 3 moves to focus search on most promising paths
        top_n = min(3, len(scored_moves))
        return [move[1] for move in scored_moves[:top_n]]
    
    def minimax(self, board, depth, alpha, beta, is_maximizing):
        self.nodes_evaluated += 1
        
        # Check for terminal state first
        if Mangala.check_terminal(board):
            return self.evaluate(board)
            
        # Check transposition table
        board_tuple = tuple(board)
        if (board_tuple, depth, is_maximizing) in self.cache:
            return self.cache[(board_tuple, depth, is_maximizing)]
            
        # Quiescence search - don't stop at max depth if the position is unstable
        available_actions = self.get_available_actions(board)
        if not available_actions:
            return self.evaluate(board)
            
        # Check max depth, but allow searching deeper in forced lines
        if depth >= self.max_depth * 2:  # Increased max depth for quiescence search
            return self.evaluate(board)
            
        if depth >= self.max_depth:
            # If there are captures or extra turns available, search deeper
            has_forced = any(Mangala.check_for_extra_turn(board, a) or 
                           self._is_capture_move(board, a) for a in available_actions)
            
            if not has_forced:
                return self.evaluate(board)
            # Else continue searching with increased depth limit
        
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
                return 10000  # Big win
            elif player_total < opponent_total:
                return -10000  # Big loss
            else:
                return 0  # Draw
        
        # 1. Store difference (most important) - heavily weighted
        store_diff = (board[player_store] - board[opponent_store]) * 5
        
        # 2. Stone count difference (material advantage)
        player_stones = sum(board[0:6])
        opponent_stones = sum(board[7:13])
        stone_diff = player_stones - opponent_stones
        
        # 3. Positional advantages
        position_value = 0
        
        # a. Extra turn potential (very valuable)
        for i in range(6):
            if board[i] == (6 - i):  # Exact landing in store
                position_value += 20  # Increased value for extra turns
        
        # b. Capture opportunities and mobility
        for i in range(6):
            # Empty pit with capture potential
            if board[i] == 0:
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
