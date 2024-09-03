import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, device=None):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device))
        else:
            self.weight = None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x

class PeepholeLSTM(nn.Module):
    def __init__(self, dim, hidden_dim, a_init_range=(1,16), dt_init_range=(0.001,0.1)):
        super().__init__()
        self.hidden_dim = hidden_dim
        dt = torch.exp(torch.empty(hidden_dim).uniform_(np.log(dt_init_range[0]), np.log(dt_init_range[1])))
        # inv_softplus_dt = torch.log(torch.exp(dt)-1) equals
        inv_softplus_dt = dt + torch.log(1-torch.exp(-dt))
        self.fc_forget = nn.Linear(dim + dim + hidden_dim, hidden_dim)
        self.fc_forget.bias = nn.Parameter(inv_softplus_dt)
        self.ln_forget_a = nn.Parameter(torch.log(torch.empty(hidden_dim).uniform_(*a_init_range)))
        self.input_fc_1 = nn.Linear(dim + dim + hidden_dim, hidden_dim) 
        self.input_fc_2 = nn.Linear(dim + dim + hidden_dim, hidden_dim) 
        self.output_fc_1 = nn.Linear(dim + dim + hidden_dim, dim) 
        self.output_fc_2 = nn.Linear(hidden_dim, dim) 
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(hidden_dim))
        self.last_out = None
        self.last_out_init = nn.Parameter(torch.randn(dim))
        self.act = nn.SiLU()
        self.is_refresh = True

    def step(self, input, last_output): # (batch, dim), (batch, dim) -> (batch, dim)
        batch, dim = input.shape
        if self.last_hidden is None:
            last_hidden = self.last_hidden_init.unsqueeze(0).expand(batch, self.hidden_dim)
        else:
            last_hidden = self.last_hidden.detach()
        x = torch.cat((input, last_output, last_hidden), -1)
        forget = torch.exp(- torch.exp(self.ln_forget_a) * F.softplus(self.fc_forget(x))) # (batch, len, hidden_dim)
        input = self.act(self.input_fc_1(x)) * self.input_fc_2(x)
        next_hidden = last_hidden * forget + input
        output = self.act(self.output_fc_1(x)) * self.output_fc_2(next_hidden)
        if self.is_refresh:
            self.last_hidden = next_hidden
        return output

    def forward(self, x):
        batch, length, dim = x.shape
        if self.last_hidden is None:
            last_out = self.last_out_init.unsqueeze(0).expand(batch, dim)
        else:
            last_out = self.last_out
        for i in range(length):
            x[:,i,:] = self.step(x[:,i,:], (last_out if i==0 else x[:,i-1,:]).detach())
        if self.is_refresh:
            self.last_out = x[:,-1,:]
        return x

    def reset_hidden(self):
        self.last_hidden = None
        self.last_out = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        if self.last_hidden is None:
            return None
        else:
            return torch.cat((self.last_hidden, self.last_out), -1)

    def set_hidden(self, hidden):
        self.last_hidden = hidden[..., :self.hidden_dim] 
        self.last_out = hidden[..., self.hidden_dim:] 

class LSTM(nn.Module):
    def __init__(self, dim, hidden_dim, a_init_range=(1,16), dt_init_range=(0.001,0.1)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(dim, hidden_dim, batch_first=True, proj_size=dim)
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(hidden_dim))
        self.last_out = None
        self.last_out_init = nn.Parameter(torch.randn(dim))
        self.act = nn.SiLU()
        self.is_refresh = True

    def forward(self, x):
        batch, length, dim = x.shape
        if self.last_hidden is None:
            last_hidden = self.last_hidden_init.unsqueeze(0).expand(batch, self.hidden_dim)
        else:
            last_hidden = self.last_hidden.detach()
        if self.last_out is None:
            last_out = self.last_out_init.unsqueeze(0).expand(batch, dim)
        else:
            last_out = self.last_out.detach()
        y, (next_out, next_hidden) = self.lstm(x, (last_out.unsqueeze(0), last_hidden.unsqueeze(0)))
        next_out = next_out.squeeze(0)
        next_hidden = next_hidden.squeeze(0)
        if self.is_refresh:
            self.last_hidden = next_hidden
            self.last_out = next_out
        return y

    def reset_hidden(self):
        self.last_hidden = None
        self.last_out = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        if self.last_hidden is None:
            return None
        else:
            return torch.cat((self.last_hidden, self.last_out), -1)

    def set_hidden(self, hidden):
        self.last_hidden = hidden[..., :self.hidden_dim] 
        self.last_out = hidden[..., self.hidden_dim:] 


class DLBBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__()
        self.plstm = LSTM(dim, hidden_dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_sconv = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.norm_sconv(x)
        x = self.plstm(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.plstm.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.plstm.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.plstm.get_hidden()

    def set_hidden(self, hidden):
        self.plstm.set_hidden(hidden)

class DLB(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_hidden: int,
        dim_ff_hidden: int,
        dropout: float,
        vocab_size: int,
        devices,
        token_in_out_parameter_corr = 3.0,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])
        self.block_list = nn.ModuleList([DLBBlock(dim, dim_hidden, dim_ff_hidden, dropout) for _ in range(depth)])
        self.norm_last = RMSNorm(dim, device=devices[-1])

        self.token_in_out_parameter_corr = token_in_out_parameter_corr
        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_norm_last = sum(p.numel() for p in self.norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)
        self.out_only_device = out_only_device

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (int)(((len(self.devices)-(1 if self.out_only_device else 0)) * (i * self.num_parameters_per_block + self.num_parameters_token_in)) / self.num_parameters)

    def forward(self, x):
        x = self.token_in(x)
        for i, block in enumerate(self.block_list):
            x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = x.to(self.devices[-1])
        x = self.norm_last(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)

    def get_hidden(self):
        hidden_list = []
        for block in self.block_list:
            hidden = block.get_hidden()
            if hidden is None:
                return None
            hidden_list.append(hidden.cpu())
        return torch.stack(hidden_list, dim=1).detach()

    def set_hidden(self, hidden_stack):
        for i, block in enumerate(self.block_list):
            block.set_hidden(hidden_stack[:,i].to(self.devices[self.device_index(i)]) if hidden_stack is not None else None)

    def module_list(self):
        blistlist = []
        for _ in self.devices:
            blistlist.append([])
        for i, block in enumerate(self.block_list):
            blistlist[self.device_index(i)].append(block)
        mlist = []
        for blist in blistlist:
            mlist.append(nn.Sequential(*blist))
        mlist[0] = nn.Sequential(self.token_in, mlist[0])
        mlist[-1] = nn.Sequential(mlist[-1], self.norm_last, self.token_out)
        return mlist
        
    