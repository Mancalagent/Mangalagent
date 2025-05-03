class MCTNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
        
    def is_terminal(self):
        return self.is_leaf() or self.is_root()
    
    def get_value(self):
        return self.value
    
    def get_visits(self):
        return self.visits
    
    def get_children(self):
        return self.children
    
    def get_state(self):
        return self.state
    
    def get_parent(self):
        return self.parent
    
    
    
    