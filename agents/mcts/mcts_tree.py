from agents.mcts import MCTNode


class MCTSTree:
    def __init__(self, **kwargs):
        self.root = MCTNode()
        self.current_node = self.root
        
        self.network = kwargs.get('network', None)
        
    def select(self, c_puct):
        
        #if root node, return None
        pass
        
        
    
    def expand(self, action, child):
        
        pass
        
    
    def simulate(self, fast_policy_network = None , game = None):
        
        pass
                
            
        
    
    def backpropagate(self, value):

        pass
            
    
    
    def get_root(self):
        return self.root
    
    
    def get_current_node(self):
        return self.current_node
    
    def run_iteration(self, c_puct):
        
           
         pass
        
        
        
        #this loop is to get the transitions of the form (state, action, game_return)
    
    
    
    
    
    
    