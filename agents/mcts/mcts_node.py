
class MCTNode:
    def __init__(self, state=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Map from action to MCTSNode
        self.visit_count = 0
        self.value_sum = 0 # sum of all returns from backprop
        self.prior_probs = None  # From policy network
        
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
        
    def get_value(self):
        return self.value_sum 
    
    def get_visits(self):
        return self.visit_count
    
    def get_children(self):
        return self.children
    
    def get_state(self):
        return self.state
    
    def get_parent(self):
        return self.parent
    
    def get_prior_probs(self):
        return self.prior_probs
    
    def get_visit_count(self):
        return self.visit_count
    
    def set_prior_probs(self, prior_probs):
        self.prior_probs = prior_probs
    
    
    
    
    
    