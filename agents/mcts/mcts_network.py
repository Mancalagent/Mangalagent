from torch import nn
from torch.nn import functional as F

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MCTS_Value_Network(nn.Module, metaclass=SingletonMeta):
    def __init__(self, input_dim = 14, output_dim = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )   
    
    def forward(self, x):
        return self.net(x)
    
    
class MCTS_Policy_Network(nn.Module, metaclass=SingletonMeta):
        def __init__(self, input_dim = 14, output_dim = 6):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            

            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
        
        def forward(self, x):
            return self.net(x)
        
        def inference(self, x):
            return F.softmax(self.forward(x), dim=1)
        
        
        