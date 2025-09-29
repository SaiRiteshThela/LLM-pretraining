import torch
import torch.nn as nn
import math
from cs336_basics.utils import my_scaled_dot_product_attention
from dataclasses import dataclass
from typing import Optional


@dataclass
class MyConfig:
    num_layers: int = 48
    d_model: int = 1600
    num_heads: int = 25
    d_ff: int = 6400
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    vocab_size:int = 50257



class MyLinear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        mean, std = 0, math.sqrt(2/(self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean, std, -3*std, 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.T
    
class MyEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.zeros(self.num_embeddings, self.embedding_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        mean ,std = 0, 1
        torch.nn.init.trunc_normal_(self.weight, mean, std, -3*std, 3*std)

    def forward(self, token_ids) -> torch.Tensor:
        return self.weight[token_ids, :]
    

class MyRMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        self.device = device if device else 'cpu'
        self.dtype = dtype if dtype else torch.float32

        self.weight = nn.Parameter(torch.zeros(self.d_model, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = 1/((x**2).mean(-1, keepdim=True) + self.eps).sqrt()
        x = self.weight * x * inv_rms
        return x.to(in_dtype)
    

class MySwiGLU(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            d_ff: int, 
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,   
        ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = MyLinear(d_model, d_ff, **factory_kwargs)
        self.w2 = MyLinear(d_ff, d_model, **factory_kwargs)
        self.w3 = MyLinear(d_model, d_ff, **factory_kwargs)

    def forward(self, x) -> torch.Tensor:
        o1 = self.w1(x)
        o1 = o1 * torch.sigmoid(o1)
        o2 = self.w3(x)
        return self.w2(o2*o1)


class MyRotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: torch.device | None = None 
    ):
        super().__init__()
        assert d_k%2 == 0, 'only even d_k supported'
        self.d_k = d_k
        cosm, sinm = self._get_trigs(max_seq_len, theta, d_k)
        self.register_buffer('cosm', cosm.to(device=device), persistent=False)
        self.register_buffer('sinm', sinm.to(device=device), persistent=False)

    def _get_trigs(self, max_seq_len: int, theta: float, d_k: int) -> tuple[torch.tensor, torch.tensor]:
        f_idx = theta**((-2.0*torch.arange(d_k//2))/d_k) #, d_k//2
        s_idx = torch.arange(max_seq_len).unsqueeze(-1) # max_seq_len, 1

        return torch.cos(f_idx * s_idx) , torch.sin(f_idx * s_idx) # max_seq_len, d_k//2
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """handle head dimension before passing to rope """
        cos_angles = self.cosm[token_positions, :] # seq_len, d_k//2
        sin_angles = self.sinm[token_positions, :] # seq_len, d_k//2

        x = x.view(*x.shape[:-1], -1,  2)
        x_even, x_odd = x[..., 0], x[..., 1]
        x_even_rot = x_even * cos_angles - x_odd * sin_angles
        x_odd_rot  = x_odd  * cos_angles + x_even * sin_angles
        return torch.stack((x_even_rot, x_odd_rot), dim=-1).view(*x.shape[:-2], self.d_k)
    
class MyCausalMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int, 
            num_heads: int,
            max_seq_len: int = 1024,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,   
    ):
        super().__init__()
        assert d_model % num_heads == 0; 'd_model should be a multiple of num_heads'
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.k_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.v_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.output_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.h_dim = self.d_model // self.num_heads
        self.max_seq_len = max_seq_len
        self.register_buffer('mask', torch.tril(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=0).unsqueeze(0).bool(), persistent=False)

    def forward(self, x: torch.tensor):
        B, T, C = x.shape

        q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, self.h_dim).transpose(-2, -3) # ... num_heads, seq_len, h_dim
        k = self.k_proj(x).view(*x.shape[:-1], self.num_heads, self.h_dim).transpose(-2, -3) # ... num_heads, seq_len, h_dim
        v = self.v_proj(x).view(*x.shape[:-1], self.num_heads, self.h_dim).transpose(-2, -3) # ... num_heads, seq_len, h_dim

        out = my_scaled_dot_product_attention(q, k, v, self.mask[:, :T, :T]).transpose(-2, -3).contiguous().view(*x.shape)
        return self.output_proj(out)
    
class MyCausalMultiHeadAttentionWithRope(nn.Module):
    def __init__(
            self,
            d_model: int, 
            num_heads: int,
            max_seq_len: int,
            theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,   
    ):
        super().__init__()
        assert d_model % num_heads == 0; 'd_model should be a multiple of num_heads'
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.k_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.v_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.output_proj = MyLinear(d_model, d_model, **factory_kwargs)
        self.theta = theta
        self.d_model = d_model
        self.num_heads = num_heads
        self.h_dim = self.d_model // self.num_heads
        self.max_seq_len = max_seq_len
        self.register_buffer('mask', torch.tril(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=0).unsqueeze(0).bool(), persistent=False)
        self.rope = MyRotaryPositionalEmbedding(self.theta, self.h_dim, max_seq_len, device=factory_kwargs['device'])

    def forward(self, x: torch.tensor, token_positions: Optional[torch.tensor] = None):
        B, T, C = x.shape
        if token_positions is None:
            token_positions = torch.arange(T, device=x.device)
        token_positions = token_positions.unsqueeze(-2)
        q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, self.h_dim).transpose(-2, -3) # ... num_heads, seq_len, h_dim
        k = self.k_proj(x).view(*x.shape[:-1], self.num_heads, self.h_dim).transpose(-2, -3) # ... num_heads, seq_len, h_dim
        v = self.v_proj(x).view(*x.shape[:-1], self.num_heads, self.h_dim).transpose(-2, -3) # ... num_heads, seq_len, h_dim
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        out = my_scaled_dot_product_attention(q, k, v, self.mask[:, :T, :T]).transpose(-2, -3).contiguous().view(*x.shape)
        return self.output_proj(out)
    

class MyTransformerBlock(nn.Module):
    def __init__(
        self,
        config: MyConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,   
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = MyRMSNorm(config.d_model, **factory_kwargs)
        self.attn = MyCausalMultiHeadAttentionWithRope(
            d_model=config.d_model, 
            num_heads=config.num_heads, 
            max_seq_len=config.max_seq_len, 
            theta=config.rope_theta,
            **factory_kwargs 
        )
        self.ffn = MySwiGLU(
            d_model=config.d_model,
            d_ff = config.d_ff,
            **factory_kwargs
        )
        self.ln2 = MyRMSNorm(config.d_model, **factory_kwargs)

    def forward(self, x: torch.tensor, token_positions: Optional[torch.tensor] = None):
        residual = x 
        x = self.attn(self.ln1(x), token_positions)
        x += residual

        residual = x
        x = self.ffn(self.ln2(x))
        x += residual

        return x
    

class MyTransformer(nn.Module):
    def __init__(
        self,
        config: MyConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.token_embeddings = MyEmbedding(config.vocab_size, config.d_model, **factory_kwargs)
        self.layers = nn.Sequential(*[MyTransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = MyRMSNorm(config.d_model)
        self.lm_head = MyLinear(config.d_model, config.vocab_size)


    def forward(self, x: torch.tensor, token_positions: Optional[torch.tensor] = None):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        
        return self.lm_head(self.ln_final(x))















    





