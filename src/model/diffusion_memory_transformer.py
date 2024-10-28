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

class DiffusionMemory(nn.Module):
    def __init__(self, num_head: int, head_dim: int, size: int, sample_size: int, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.sample_size = sample_size
        self.num_head = num_head
        self.head_dim = head_dim
        self.size = size
        self.key_store = nn.Parameter(torch.randn(size, num_head, head_dim), requires_grad=False)
        self.value_store = nn.Parameter(torch.randn(size, num_head, head_dim), requires_grad=False)
        self.logit_store = nn.Parameter(torch.zeros(size, num_head), requires_grad=False)

    def read(self, query):
        with torch.no_grad():
            batch, num_head, head_dim = query.shape
            size = self.size
            sample_size = self.sample_size
            logit_store_batch = self.logit_store.unsqueeze(0).expand(batch, size, num_head)
            softmax_sample = nn.functional.softmax(logit_store_batch.transpose(2,1).reshape(batch * num_head, size), dim=0)
            indices = torch.multinomial(softmax_sample, sample_size).reshape(batch, num_head, sample_size).transpose(2,1) # (batch, sample_size, num_head)
            sample_keys = torch.gather(self.key_store.unsqueeze(0).expand(batch, size, num_head, head_dim), 1, indices.unsqueeze(3)) # (batch, sample_size, num_head, head_dim)
            sample_values = torch.gather(self.value_store.unsqueeze(0).expand(batch, size, num_head, head_dim), 1, indices.unsqueeze(3)) # (batch, sample_size, num_head, head_dim)
            softmax_qk = nn.functional.softmax(torch.einsum("bhd,bshd->bsh", query, sample_keys) * (head_dim ** -0.5), dim=1)
            lerp_key = torch.einsum("bsh,bshd->bhd", softmax_qk, sample_keys)
            lerp_value = torch.einsum("bsh,bshd->bhd", softmax_qk, sample_values)
            batch_index, head_index = torch.meshgrid(torch.arange(batch), torch.arange(num_head), indexing="ij")
            self.key_store[indices[batch_index,:,head_index], head_index.unsqueeze(2), :] = torch.einsum("bhs,bhsd->bhsd", 1-softmax_qk[batch_index,:,head_index], sample_keys[batch_index,:,head_index,:]) + torch.einsum("bhs,bhd->bhsd", softmax_qk[batch_index,:,head_index], lerp_key[batch_index, head_index])
            self.value_store[indices[batch_index,:,head_index], head_index.unsqueeze(2), :] = torch.einsum("bhs,bhsd->bhsd", 1-softmax_qk[batch_index,:,head_index], sample_values[batch_index,:,head_index,:]) + torch.einsum("bhs,bhd->bhsd", softmax_qk[batch_index,:,head_index], lerp_value[batch_index, head_index])
            self.logit_store[indices[batch_index,:,head_index], head_index.unsqueeze(2)] = self.gamma * self.logit_store[indices[batch_index,:,head_index], head_index.unsqueeze(2)] + softmax_qk[batch_index,:,head_index]
            return lerp_key, lerp_value

    def write(self, key, value):
        with torch.no_grad():
            batch, num_head, head_dim = key.shape
            size = self.size
            sample_size = self.sample_size
            logit_store_batch = self.logit_store.unsqueeze(0).expand(batch, size, num_head)
            softmax_sample = nn.functional.softmax(-logit_store_batch.transpose(2,1).reshape(batch * num_head, size), dim=0)
            indices = torch.multinomial(softmax_sample, sample_size).reshape(batch, num_head, sample_size).transpose(2,1) # (batch, sample_size, num_head)
            sample_keys = torch.gather(self.key_store.unsqueeze(0).expand(batch, size, num_head, head_dim), 1, indices.unsqueeze(3)) # (batch, sample_size, num_head, head_dim)
            sample_values = torch.gather(self.value_store.unsqueeze(0).expand(batch, size, num_head, head_dim), 1, indices.unsqueeze(3)) # (batch, sample_size, num_head, head_dim)
            softmax_kk = nn.functional.softmax(torch.einsum("bhd,bshd->bsh", key, sample_keys) * (head_dim ** -0.5), dim=1)
            batch_index, head_index = torch.meshgrid(torch.arange(batch), torch.arange(num_head), indexing="ij")
            self.key_store[indices[batch_index,:,head_index], head_index.unsqueeze(2), :] = torch.einsum("bhs,bhsd->bhsd", 1-softmax_kk[batch_index,:,head_index], sample_keys[batch_index,:,head_index,:]) + torch.einsum("bhs,bhd->bhsd", softmax_kk[batch_index,:,head_index], key[batch_index, head_index])
            self.value_store[indices[batch_index,:,head_index], head_index.unsqueeze(2), :] = torch.einsum("bhs,bhsd->bhsd", 1-softmax_kk[batch_index,:,head_index], sample_values[batch_index,:,head_index,:]) + torch.einsum("bhs,bhd->bhsd", softmax_kk[batch_index,:,head_index], value[batch_index, head_index])
            #self.logit_store[indices[batch_index,:,head_index], head_index.unsqueeze(2)] = self.gamma * self.logit_store[indices[batch_index,:,head_index], head_index.unsqueeze(2)] + softmax_kk[batch_index,:,head_index]
            self.logit_store[indices[batch_index,:,head_index], head_index.unsqueeze(2)] = softmax_kk[batch_index,:,head_index] / (1-self.gamma)



class DMMHA(nn.Module):
    def __init__(self, dim: int, num_head: int, band_width: int, mem_size: int, mem_sample: int, mem_gamma: int, is_memory: bool, is_bias: bool=False):
        super().__init__()
        assert dim % num_head == 0, "dim must be multiple of num_head"
        self.dim = dim
        self.num_head = num_head
        self.band_width = band_width
        head_dim = dim // num_head

        self.linear_q = nn.Linear(dim, dim, bias=is_bias)
        self.linear_k = nn.Linear(dim, dim, bias=is_bias)
        self.linear_v = nn.Linear(dim, dim, bias=is_bias)
        self.linear_out = nn.Linear(dim, dim, bias=is_bias)
        self.memory = DiffusionMemory(num_head, head_dim, mem_size, mem_sample, mem_gamma) if is_memory else None
        self.posemb = nn.Parameter(torch.randn(band_width * 2, num_head, head_dim))

        self.last_q = None
        self.last_q_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))
        self.last_k = None
        self.last_k_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))
        self.last_v = None
        self.last_v_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))
        self.last_mk = None
        self.last_mk_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))
        self.last_mv = None
        self.last_mv_init = nn.Parameter(torch.randn(band_width-1, num_head, head_dim))

        self.is_refresh = True

    def forward(self, x):
        batch, length, dim = x.shape
        assert dim == self.dim, "dim missmatch"
        num_head = self.num_head
        head_dim = dim // num_head
        band_width = self.band_width

        q = self.linear_q(x).view(batch, length, num_head, head_dim)
        k = self.linear_k(x).view(batch, length, num_head, head_dim)
        v = self.linear_v(x).view(batch, length, num_head, head_dim)
        mk = torch.zeros(batch, length, num_head, head_dim, device=x.device, dtype=x.dtype)
        mv = torch.zeros(batch, length, num_head, head_dim, device=x.device, dtype=x.dtype)

        if self.last_q is None:
            last_q = self.last_q_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
            last_k = self.last_k_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
            last_v = self.last_v_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
            last_mk = self.last_mk_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
            last_mv = self.last_mv_init.unsqueeze(0).expand(batch, band_width-1, num_head, head_dim)
        else:
            last_q = self.last_q.detach()
            last_k = self.last_k.detach()
            last_v = self.last_v.detach()
            last_mk = self.last_mk.detach()
            last_mv = self.last_mv.detach()
        q_with_last_q = torch.cat((last_q, q), dim=1)
        k_with_last_k = torch.cat((last_k, k), dim=1)
        v_with_last_v = torch.cat((last_v, v), dim=1)
        mk_with_last_mk = torch.cat((last_mk, mk), dim=1)
        mv_with_last_mv = torch.cat((last_mv, mv), dim=1)

        if self.memory is not None:
            with torch.no_grad():
                for i in range(length):
                    mk_with_last_mk[:,i + band_width - 1,:,:], mv_with_last_mv[:,i + band_width-1,:,:] = self.memory.read(q_with_last_q[:,i,:,:])
                    self.memory.write(mk_with_last_mk[:,i,:,:], mv_with_last_mv[:,i,:,:])
        mk_with_last_mk = mk_with_last_mk.detach()
        mv_with_last_mv = mv_with_last_mv.detach()

        memory_attention_matrix = torch.zeros(batch, length, band_width, num_head, device=x.device, dtype=x.dtype)
        for i in range(band_width):
            memory_attention_matrix[:,:,i,:] = torch.einsum("blhd,blhd->blh", q, mk_with_last_mk[:,i:i+length,:,:])
        memory_attention_matrix *= head_dim ** -0.5

        current_attention_matrix = torch.zeros(batch, length, band_width, num_head, device=x.device, dtype=x.dtype)
        for i in range(band_width):
            current_attention_matrix[:,:,i,:] = torch.einsum("blhd,blhd->blh", q, k_with_last_k[:,i:i+length,:,:])
        current_attention_matrix *= head_dim ** -0.5

        attention_matrix = torch.cat((memory_attention_matrix, current_attention_matrix), dim=2)
        attention_matrix = nn.functional.softmax(attention_matrix, dim=2)
        out_acc = torch.zeros(batch, length, num_head, head_dim, device=x.device, dtype=x.dtype)
        for i in range(band_width):
            out_acc += torch.einsum("blh,blhd->blhd", attention_matrix[:,:,i,:], v_with_last_v[:,i:i+length,:,:] + self.posemb[i].unsqueeze(0).unsqueeze(1))
        for i in range(band_width):
            out_acc += torch.einsum("blh,blhd->blhd", attention_matrix[:,:,band_width+i,:], mv_with_last_mv[:,i:i+length,:,:] + self.posemb[band_width+i].unsqueeze(0).unsqueeze(1))
        out = self.linear_out(out_acc.reshape(batch, length, dim))

        if self.is_refresh:
            self.last_q = q_with_last_q[:,-(self.band_width-1):,:,:]
            self.last_k = k_with_last_k[:,-(self.band_width-1):,:,:]
            self.last_v = v_with_last_v[:,-(self.band_width-1):,:,:]
            self.last_mk = mk_with_last_mk[:,-(self.band_width-1):,:,:]
            self.last_mv = mv_with_last_mv[:,-(self.band_width-1):,:,:]

        return out

    def reset_hidden(self):
        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_mk = None
        self.last_mv = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

    def get_hidden(self):
        if self.last_q is None:
            return None
        return torch.stack((self.last_q, self.last_k, self.last_v, self.last_mk, self.last_mv))

    def set_hidden(self, hidden):
        self.last_q = hidden[0]
        self.last_k = hidden[1]
        self.last_v = hidden[2]
        self.last_mk = hidden[3]
        self.last_mv = hidden[4]
        
class DMTBlock(nn.Module):
    def __init__(self, dim: int, num_head: int, dim_ff_hidden: int, band_width: int, mem_size: int, mem_sample: int, mem_gamma: int, dropout: float, is_memory: bool):
        super().__init__()
        self.band_mha = DMMHA(dim, num_head, band_width, mem_size, mem_sample, mem_gamma, is_memory)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
        self.norm_band_mha = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.norm_band_mha(x)
        x = self.band_mha(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.band_mha.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.band_mha.set_is_refresh(is_refresh)

    def get_hidden(self):
        return self.band_mha.get_hidden()

    def set_hidden(self, hidden):
        self.band_mha.set_hidden(hidden)

class DiffusionMemoryTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_head: int,
        dim_ff_hidden: int,
        mem_size: int,
        mem_sample: int,
        mem_gamma: float,
        dropout: float,
        vocab_size: int,
        devices,
        band_width: int,
        token_in_out_parameter_corr = 3.0,
        out_only_device: bool=True,
    ):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Embedding(vocab_size, dim, device=devices[0], max_norm=1)
        self.block_list = nn.ModuleList([DMTBlock(dim, num_head, dim_ff_hidden, band_width, mem_size, mem_sample, mem_gamma, dropout, i == depth // 2) for i in range(depth)])
        self.norm_last = RMSNorm(dim, device=devices[-1])
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1])

        self.token_in_out_parameter_corr = token_in_out_parameter_corr
        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_norm_last = sum(p.numel() for p in self.norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = self.num_parameters_token_in + (self.num_parameters_per_block * depth) + self.num_parameters_norm_last + self.num_parameters_token_out
        self.out_only_device = out_only_device

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        num_out_params = self.num_parameters_norm_last + self.num_parameters_token_out
        return (int)(
            (
                (len(self.devices)-(1 if self.out_only_device else 0)) *
                (i * self.num_parameters_per_block + self.num_parameters_token_in)
            ) /
            (self.num_parameters - (num_out_params if self.out_only_device else 0))
        )

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
        
    