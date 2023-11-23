import torch
import torch.nn as nn
import numpy as np

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_1.weight, std=dim**-0.5)
        nn.init.constant_(self.linear_1.bias, 0)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_2.weight, std=dim_ff_hidden**-0.5)
        nn.init.constant_(self.linear_2.bias, 0)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

class SConv(nn.Module):
    def __init__(self, dim: int, dtype):
        super().__init__()
        self.dim = dim
        self.phazor = nn.Parameter(torch.exp(2.0j * np.pi * torch.arange(dim) / dim) * torch.abs(torch.randn(dim)))
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.is_refresh = True


    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dtype = x.dtype

        x = x.to(torch.cfloat)
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, self.dim)
        else:
            self.last_conv = self.last_conv.detach()
        phazor = self.phazor
        phazor = torch.tanh(phazor.abs()) * torch.exp(1.0j * phazor.angle())
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)) # (len, dim)
        filter = phazor_progression# * phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len*2, dim=1) # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1,0,len) # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:]

        y = conv_with_past
        y = y.real.to(dtype)
        return y

    def reset_hidden(self):
        self.last_conv = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SConvNetBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 
        self.spiral_conv = SConv(dim, dtype)
        self.ffn = FFN(dim, dim_ff_hidden, dtype)
        self.layer_norm_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.layer_norm_ffn_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.sc_elementwise_linear = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm_sc_in(x)
        x = self.act(x)
        x = self.spiral_conv(x)
        x = x * self.sc_elementwise_linear
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm_ffn_in(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.spiral_conv.set_is_refresh(is_refresh)

class SConvNet(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        dropout: float,
        vocab_size: int,
        devices,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.token_in = nn.Linear(vocab_size, dim, device=devices[0], dtype=dtype)
        nn.init.normal_(self.token_in.weight, std=vocab_size**-0.5)
        nn.init.constant_(self.token_in.bias, 0)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        nn.init.normal_(self.token_out.weight, std=dim**-0.5)
        nn.init.constant_(self.token_out.bias, 0)
        self.block_list = nn.ModuleList([SConvNetBlock(dim, dim_ff_hidden, dropout, dtype) for _ in range(depth)])
        self.layer_norm_last = nn.LayerNorm(dim, elementwise_affine=True, bias=True, device=devices[-1], dtype=dtype)
        for i, block in enumerate(self.block_list):
            block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (len(self.devices) * i) // len(self.block_list)

    def forward(self, x):
        x = self.token_in(x)
        for i, block in enumerate(self.block_list):
            if i > 0 and self.device_index(i) != self.device_index(i-1):
                x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = self.layer_norm_last(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)

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
        mlist[-1] = nn.Sequential(mlist[-1], self.layer_norm_last, self.token_out)
        return mlist
        
    