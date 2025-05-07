from torch import nn
from torch.nn import functional as F

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MCTSNetwork(nn.Module, metaclass=SingletonMeta):
    """ 
    This network is for learning one model with two heads 
    One head is for the value network and the other is for the policy network
    """
    def __init__(self):
        super().__init__()
        
        
        
    def forward(self, x):
        return x


class MCTSValueNetwork(nn.Module, metaclass=SingletonMeta):
    """
    Value network takes a state and returns a value regarding that state
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
    
class MCTSPolicyNetwork(nn.Module, metaclass=SingletonMeta):
    """
    Policy network takes a state and returns a policy
    This network is a REINFORCE policy gradient network 
    
    if Supervised network is used:
          initilized with weights from the supervised network
    else: 
          initilized with random weights
    """
    
    def __init__(self, supervised_network=None):
        super().__init__()
        
        if supervised_network is not None:
            raise NotImplementedError("Supervised network not implemented")
        else:
            pass
                

    def forward(self, x):
        return x
    
    
    
    
class MCTSFastPolicyNetwork(nn.Module, metaclass=SingletonMeta):
    
    """
    This network is used for fast rollouts, which takes a state and returns a policy
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
    
class MCTSSupervisedNetwork(nn.Module, metaclass=SingletonMeta):
    """
    Supervised network train on passed plays
    This network takes a state and returns a policy
    
    This network is finetuned before the policy network
    
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
        
        
        