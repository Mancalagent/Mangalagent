import base64
import json
import pickle

from agents.mcts.mcts_node import MCTSNode, State
from agents.random_agent import RandomAgent
from mangala.mangala import Mangala


class MCTSTree:
    def __init__(self, **kwargs):
        init_state = kwargs.get('init_state', None)
        # print("initial state", init_state)
        
        self.root = MCTSNode(init_state)
        # print("root", self.root)
        
        
        self.policy_network = kwargs.get('policy_network', None)
        self.value_network = kwargs.get('value_network', None)
        
    
        
    def run_iteration(self, c_puct):
        current_node = self.root
        trajectories = []
        best_child = self._select(current_node, c_puct)
    
        
        
        if best_child == -1:
            # print(f"best child is terminal state {best_child.get_state().get_state()}")
            
            r = current_node.get_outcome()
            # print("hit terminal child with outcome", r)
            trajectories = self._backpropagate(r, best_child)
            return trajectories
        
        elif best_child is None:
            # print(f"best child is root {current_node.get_state().get_state()}")
            self._expand(current_node)
            for child in current_node.get_children():
                r = self._simulate(child)
                trajectories.append(self._backpropagate(r, child))
            return trajectories
            
        else:
            # print(f"best child is {best_child.get_state().get_state()}")
            
            self._expand(best_child)
            for child in best_child.get_children():
                r = self._simulate(child)
                trajectories.append(self._backpropagate(r, child))
            return trajectories
            
                

        
        
            
    
    def _select(self, current_node, c_puct):
        """
        Select the best child node according to the PUCT formula.
        
        Args:
            current_node: The node to select from
            c_puct: Exploration constant
            
        Returns:
            (path, node): Tuple with the path taken and the final leaf/terminal node
        """
        # print("selecting")

        
        while True:
            # If node is terminal, return special value -1 
            if current_node.is_terminal():
                return  -1
            
            # If node is a leaf, return None to indicate expansion needed
            if current_node.is_leaf():
                return  None
                
            # If node has children, select best child
            best_score = float('-inf')
            best_child = None
            
            # Calculate the sum of visit counts across all edges
            total_edge_visits = sum(edge.get_visits() for edge in current_node.get_edges())
          
            # Iterate through all edges to find the best child according to PUCT
            for edge in current_node.get_edges():
                child = edge.get_target()
                
                # Set prior probability if not already set
                prior_prob = edge.get_prior_prob()
                
                # Standard PUCT formula
                # For unvisited nodes, Q-value is 0
                Q = 0
                if edge.get_visits() > 0:
                    Q = edge.get_value() / edge.get_visits()
                    
                # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
                U = c_puct * prior_prob * (total_edge_visits ** 0.5) / (1 + edge.get_visits())
                
                # print(f"action {edge.get_action()} has Q + U value {Q + U}, with prior probability {prior_prob} and value {edge.get_value()} and visits {edge.get_visits()}")
                score = Q + U
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            # If couldn't find a child (shouldn't happen), return current node
            if best_child is None:
                return  current_node
                
            # Move to the best child and continue search
            current_node = best_child

            
            # If this child is a leaf or terminal, return it
            if current_node.is_leaf() or current_node.is_terminal():
                return  current_node if not current_node.is_terminal() else -1
        
    def _expand(self, current_node):
        
        # if self.policy_network:
        #     pass

                
        # else:
        # print("expanding to all possible children (UNIFORM PRIOR PROB)")
        current_node.add_available_childen()
        # print("parent edge", current_node.get_parent_edge())
        
    def _simulate(self,current_node):
        # print("simulating")
        state = current_node.get_state()
        game = Mangala(RandomAgent("player1"), RandomAgent("player2"), state.get_state())
        game.start()
        if game.get_winner() == 0:
            return 1
        elif game.get_winner() == 1:
            return -1
        else:
            return 0
        
    def _backpropagate(self, value, node):
        trajectory = []
        # print("backpropagating")

        while node.get_parent_edge() is not None:
            trajectory.append((node.get_parent().get_state(),node.get_parent_edge().get_action(), value))
            edge = node.get_parent_edge()
            edge.update_value(value)
            edge.update_visits()
            node = edge.get_source()
                
        return trajectory
        
    def get_root(self):
        return self.root
    
    
    def get_current_node(self):
        return self.current_node

    def visualize_tree(self, node=None, depth=0, max_depth=3, action_taken=None):
        """
        Visualize the MCTS tree structure with board states, actions, and statistics.
        
        Args:
            node: Starting node to visualize (default: root)
            depth: Current depth in tree for indentation
            max_depth: Maximum depth to visualize to prevent overwhelming output
            action_taken: Action that led to this node (None for root)
        """
        if node is None:
            node = self.root
            print("\n==== MCTS TREE VISUALIZATION ====")
        
        # Create indentation based on depth
        indent = "    " * depth
        
        # Print node information
        print(f"{indent}Node (depth {depth}):")
        
        # Show the action that led to this node
        if action_taken is not None:
            print(f"{indent}├── Action: {action_taken}")
        
        # Show the board state
        try:
            state_repr = str(node.get_state().get_state())
        except:
            # Fallback if get_state() doesn't return a string-convertible object
            state_repr = "State object"
            
        if len(state_repr) > 100:  # Truncate long state representations
            state_repr = state_repr[:97] + "..."
        print(f"{indent}├── State: {state_repr}")
        
        # Show if node is terminal
        print(f"{indent}├── Terminal: {node.is_terminal()}")
        
        # Show available edges and their statistics
        edges = node.get_edges()
        
        if not edges:
            print(f"{indent}└── No children")
            return
            
        print(f"{indent}└── Children: {len(edges)}")
        
        # Stop if we've reached max depth
        if depth >= max_depth:
            print(f"{indent}    └── (max depth reached)")
            return
        
        # Sort edges by visits for more meaningful visualization
        sorted_edges = sorted(edges, key=lambda e: e.get_visits(), reverse=True)
        
        # Print children
        for i, edge in enumerate(sorted_edges):
            is_last = (i == len(sorted_edges) - 1)
            
            # Edge statistics
            child = edge.get_target()
            visits = edge.get_visits()
            value = edge.get_value()
            action = edge.get_action()
            
            # Use different connector for last child
            connector = "└── " if is_last else "├── "
            child_indent = indent + ("    " if is_last else "│   ")
            
            print(f"{indent}{connector}Edge {action} (visits: {visits}, value: {value:.2f}, Q: {value/max(1,visits):.2f})")
            
            # Recursively visualize child
            self.visualize_tree(child, depth + 1, max_depth, action)

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
    def load_tree(cls, file_path='mcts_tree.json'):
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




if __name__ == "__main__":
    # initilize board 
    board = [4] * 14
    board[6] = 0
    board[13] = 0
    
    print("initial board", board)
    state = State(board)
    
    trajectories_data = []
    
    tree = MCTSTree(init_state=state)
    for i in range(10):
        print(f"iteration {i}")
        trajectories = tree.run_iteration(c_puct=1.0)
        trajectories_data.append(trajectories)
   
    # tree.visualize_tree(tree.get_root(), 0, 3)
    
    
    
    
    
    