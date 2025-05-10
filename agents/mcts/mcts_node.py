from mangala.mangala import Mangala

class State:
    
    def __init__(self, board):
        self.board = board
        
    def get_state(self):
        return self.board
    
    def get_legal_actions(self):
        """
        Returns the legal actions for the current state
        
        returns: [0, 1, 2, 3, 4, 5] if all actions are legal
        """
        return [i for i in range(len(self.board[:6])) if self.board[i] != 0]
    
class Edge:
    def __init__(self, action: int,source=None,target=None, prior_prob=0.0):
        self.action = action
        self.source = source #parent node
        self.target = target #child node
        
        self.visits = 0
        self.value = 0
        self.prior_prob = prior_prob
        
    def update_value(self, value: float):
        self.value += value
    
    def update_visits(self):
        self.visits += 1
        
    def get_value(self):
        return self.value
    
    
    def get_visits(self):
        return self.visits
    
    def get_action(self):
        return self.action
    
    
    def get_source(self):
        return self.source
    
    def get_target(self):
        return self.target
    
    def get_prior_prob(self):
        return self.prior_prob
    
    def set_prior_prob(self, prior_prob: float):
        self.prior_prob = prior_prob

    
    
    
    
    
class MCTSNode:
    def __init__(self, state: State, parent=None, parent_edge=None):
        self.state = state
        self.parent = parent
        self.parent_edge = parent_edge
        self.edges = []
        self.network = None
        
        if Mangala.check_terminal(self.state.get_state()):
            self.terminal = True
        else:
            self.terminal = False
            
    def add_edge(self, edge: Edge):
        self.edges.append(edge)
    
    def get_edges(self):
        return self.edges
    
    def get_state(self):
        """
        returns the state of the node
        
        returns: State
        """
        return self.state
    
    def get_parent(self):
        return self.parent
    
    def get_children(self):
        return [edge.target for edge in self.edges]
    
    def is_leaf(self):
        return len(self.edges) == 0
    
    
    def add_available_childen(self):
        """
        Add legal action children directly to the node
        
        returns: None
        """
        for action in self.state.get_legal_actions():
            # print(f"Adding child for action: {action}")
            # print(self.state.get_state())
            child, _, _ = Mangala.transition(self.state.get_state(), action)
            # print(child)
            child = MCTSNode(State(child), self)
            if self.network:
                pass 
            else:
                edge = Edge(action, self, child, prior_prob=1.0 / len(self.state.get_legal_actions()))
                child.set_parent_edge(edge)
                self.add_edge(edge)
    
    def is_terminal(self):
        return self.terminal
    
    def set_parent_edge(self, parent_edge: Edge):
        self.parent_edge = parent_edge
        
    def get_parent_edge(self):
        return self.parent_edge
    
    
    
    
    
    #########################################################
    #these methods are for getting the total visits and value of the node from edges
    def get_total_visits(self):
        return sum([edge.get_visits() for edge in self.edges])
    
    def get_total_value(self):
        return sum([edge.get_value() for edge in self.edges])
    
    #########################################################
    
        
        
if __name__ == "__main__":
    
    # print("Testing State class")
    state = State([1, 0, 1, 1, 0, 1, 1,\
                   0, 1, 0, 0, 0, 0, 0, ])
    
    # print(state.get_legal_actions())
    
    # print("Testing Edge class")
    # edge = Edge(0, state, state)
    # print(edge.get_action())
    # print(edge.get_source().get_state())
    # print(edge.get_target().get_state())
    
    # print("Testing MCTSNode class")
    node = MCTSNode(state)
    node.add_available_childen()
    for child in node.get_children():
        print(child.get_state())
    
    
    