import torch
import math

def my_softmax(x: torch.tensor, dim: int) -> torch.tensor:
    assert dim < x.ndim; "input a valid dim"
    max_v, i = x.max(dim=dim, keepdim=True)
    x = x - max_v
    x = x.exp()
    return x/x.sum(dim=dim, keepdim=True)

def my_silu(x: torch.tensor) -> torch.tensor:
    return x.sigmoid() * x

def my_scaled_dot_product_attention(Q: torch.tensor, K:torch.tensor, V: torch.tensor, mask: torch.tensor) -> torch.tensor:
    d_model = Q.shape[-1]
    attn = Q @ K.transpose(-1, -2) / math.sqrt(d_model)
    attn.masked_fill_(~mask, float('-inf'))
    attn_wei = my_softmax(attn, dim=-1)

    return attn_wei @ V

def my_cross_entropy_loss(logits: torch.tensor, labels):
    logits = logits.flatten(0, -2)
    labels = labels.flatten()
    max_logit, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - max_logit
    logprob = logits - logits.exp().sum(dim=-1, keepdim=True).log()
    nll = -logprob[torch.arange(labels.size(0)), labels]


    return nll.mean()    


def my_cosine_learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    if t < t_w:
        return (t/t_w)*lr_max
    if t >= t_w and t <= t_c:
        return lr_min + 0.5 * (1 + math.cos(((t - t_w)/(t_c - t_w))*math.pi))*(lr_max - lr_min)
    else:
        return lr_min

import torch

@torch.no_grad()
def my_gradient_clipping(params, max_norm, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0

    total_norm = torch.linalg.vector_norm(torch.stack([g.norm(2) for g in grads]), ord=2)

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)

    return float(total_norm)



def my_save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out)

def my_load_checkpoint(p, m, opt=None):
    import torch
    c = torch.load(p)
    sd = c["model"]
    if sd and next(iter(sd)).startswith("_orig_mod."):
        sd = {k[10:]: v for k, v in sd.items()}  # strip "_orig_mod."
    m.load_state_dict(sd, strict=False)
    if opt and "optimizer" in c:
        try: opt.load_state_dict(c["optimizer"])
        except: pass
    return c.get("iteration")



@torch.no_grad()
def generate_seq(model, start_seq, max_gen_len=100, num_samples=4, p=1.0, temperature=1.0, device="cpu"):
    model.eval()
    x = start_seq.to(device).long()
    if x.dim() == 1: x = x[None, :]
    x = x.repeat(num_samples, 1)

    while x.size(1) < max_gen_len:
        with torch.no_grad():
            logits = model(x)[:, -1] / max(temperature, 1e-8)   # (B, V)
        probs = logits.softmax(-1)
        sp, si = probs.sort(-1, descending=True)
        cp = sp.cumsum(-1)
        sp[cp > p] = 0
        sp /= sp.sum(-1, keepdim=True)
        next_tok = torch.multinomial(sp, 1)
        next_idx = si.gather(-1, next_tok)
        x = torch.cat([x, next_idx], 1)
    return x
