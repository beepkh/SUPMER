import torch
from torch import nn
from torch.nn import functional as F

def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s

class GradientTransform(nn.Module):
    def __init__(self, 
                 grad_dim:int,
                 n_hidden: int,
                 rank:int=None,
                 init: str="id",
                 norm: bool=False,
                 choice: int=0):
        super().__init__()
        self.norm = norm
        self.norm_init = False
        self.register_buffer("grad_mean", torch.full((grad_dim,), float("nan")))
        self.register_buffer("grad_std", torch.full((grad_dim,), float("nan")))
        self.register_buffer("grad_s", torch.full((grad_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))
        self.choice = choice
        self.mlp = IDMLP(grad_dim, n_hidden, rank, init, choice=choice)
    
    def reg_loss(self, sim1, sim2, coeff):
        sim = 0.5 * sim1 + 0.5 * sim2
        return (sim-coeff) ** 2
    
    def forward(self, grad, embed):
        grad = grad.to(torch.float32)
        grad_ = grad.view(-1, grad.shape[-1])
        
        if self.training:
            for idx in range(grad_.shape[0]):
                if not self.norm_init:
                    self.grad_mean = grad_[idx].clone().detach()
                    self.grad_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(grad_[idx], self.grad_mean, self.grad_s, self.k)
            self.grad_std = (self.grad_s / (self.k - 1)) ** 0.5
        
        if self.norm:
            grad_input = (grad_ - self.grad_mean) / (self.grad_std + 1e-7)
        else:
            grad_input = grad_
        
        return self.mlp(grad_input, embed)

    
class IDMLP(nn.Module):
    def __init__(self,
                 indim:int,
                 n_hidden: int,
                 rank:int=None,
                 init: str="id",
                 hiddim:int=24,
                 choice:int=0
                 ):
        super().__init__()
        self.choice = choice
        self.linear_1 = nn.Linear(indim, hiddim)
        self.linear_2 = nn.Linear(hiddim, 1)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList(
            [
                LRLinear(indim, indim, rank=rank, init=init)
                for idx in range(n_hidden + 1)
            ]
        )
    
    def forward(self, x, embed):
        x0 = x
        for layer in self.layers:
            x = layer(x)
        ### Manhattan distance 1 ###
        if self.choice==3:
            x = x / torch.mean(torch.abs(x)) * torch.mean(torch.abs(x0))
        ### Manhattan distance 2 ###
        elif self.choice==2:
            x_mean = torch.mean(torch.abs(x), 1).unsqueeze(1)
            x_mean_0 = torch.mean(torch.abs(x0), 1).unsqueeze(1)
            x = x / x_mean * x_mean_0
        ### Euclidean distance 1 ###
        elif self.choice==1:
            x_mean = torch.sum(x ** 2) ** 0.5
            x_mean_0 = torch.sum(x0 ** 2) ** 0.5
            x = x / x_mean * x_mean_0
        ### Euclidean distance 2 ###
        elif self.choice == 0:
            x_mean = torch.sum(x ** 2, 1) ** 0.5 + 1e-8
            x_mean_0 = torch.sum(x0 ** 2, 1) ** 0.5 + 1e-8
            x = x / torch.unsqueeze(x_mean, 1) * torch.unsqueeze(x_mean_0, 1)
        
        coeff = self.get_coeff(embed)
        return (1-coeff)*x + coeff*x0, coeff
 
 
    def get_coeff(self, embed):
        embed1 = self.linear_1(embed)
        embed1 = self.relu(embed1)
        embed2 = self.linear_2(embed1)
        return F.sigmoid(embed2)
 
    
    def reg_loss(self, sim1, sim2, coeff):
        sim = 0.5 * sim1 + 0.5 * sim2
        return (sim-coeff) ** 2
        
               

class LRLinear(nn.Module):
    def __init__(self, inf, outf, rank: int = None, init="id"):
        super().__init__()
        mid_dim = min(rank, inf)
        if init == "id":
            self.u = nn.Parameter(torch.randn(outf, mid_dim))
            self.v = nn.Parameter(torch.randn(mid_dim, inf))
        elif init == "xavier":
            self.u = nn.Parameter(torch.empty(outf, mid_dim))
            self.v = nn.Parameter(torch.empty(mid_dim, inf))
            nn.init.xavier_uniform_(self.u.data, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.v.data, gain=1.0)
        else:
            raise ValueError(f"Unrecognized initialization {init}")
        self.bias = nn.Parameter(torch.zeros(outf))
        self.inf = inf
        self.init = init
        
    def forward(self, x):
        pre_act = (self.u @ (self.v @ x.T)).T
        if self.bias is not None:
            pre_act += self.bias
        
        acts = pre_act.clamp(min=0)
        return acts
        
        
        
            
        