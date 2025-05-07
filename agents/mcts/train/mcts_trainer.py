from mcts.mcts_tree import MCTSTree


class MCTSTrainer:
    def __init__(self, **kwargs):
        
        self.game_count = kwargs.get('game_count', 1000)
        self.policy_network = kwargs.get('policy_network', None)
        self.value_network = kwargs.get('value_network', None)
        self.tree = MCTSTree(policy_network=self.policy_network, value_network=self.value_network)
        
        
        
    def train(self, **kwargs):
        
        for game in range(self.game_count):
            iter_path = self.tree.run_iteration(kwargs.get('c_puct', 1.0))
            
            if iter_path is not None:
                self.tree.expand(iter_path[0], iter_path[1])
            
            
            
            
            
           
           
           
        
        
        
        
        
        
        
        
        

