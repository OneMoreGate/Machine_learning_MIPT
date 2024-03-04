from torch import nn

def create_model():
    model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 16),
    nn.ReLU(),
    nn.Linear(16, 10)    
    )
    return model

def count_parameters(model):
    tot_par = sum([p.numel() for p in model.parameters()])    
    return tot_par
    