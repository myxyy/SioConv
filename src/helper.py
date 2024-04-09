import torch

def calc_num_parameters(model):
    return sum(p.numel() * (2 if torch.is_complex(p) else 1) for p in model.parameters() if p.requires_grad)