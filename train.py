print('running training')
# float32_max precision, autocast, compile, sdpa, weight decay only 2d tensor, fused = True
import os
import torch
import torch.optim as optim
from dataclasses import dataclass
from cs336_basics.layers import MyConfig, MyTransformer
from cs336_basics.data_loader import MyDataLoader
from cs336_basics.utils import my_cosine_learning_rate_schedule, generate_seq, my_save_checkpoint
import torch.nn.functional as F
import random
import numpy as np
import time
from cs336_basics.bpe_tokenizer import BPETokenizer

import wandb

learning_rate = 3e-4
max_seq_len = 256
batch_size = 64
max_lr = 3e-4
min_lr = max_lr*0.1
warmups = 684
max_steps = 34195
print_steps = 300

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- NEW: initialize wandb run ---
run_id = os.getenv("WANDB_RUN_ID")  
wandb.init(
    project=os.getenv("WANDB_PROJECT", "llm-pretraining"),
    name=os.getenv("WANDB_NAME", None),
    id=run_id,
    resume="allow" if run_id else None,
    config={
        "seed": SEED,
        "learning_rate": learning_rate,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "optimizer": "AdamW(fused)",
        "model": {
            "num_layers": 4, "d_model": 512, "num_heads": 16, "d_ff": 1344,
            "max_seq_len": max_seq_len, "rope_theta": 10000.0, "vocab_size": 10000
        },
        "precision": "bfloat16 autocast",
        "scheduler": "cosine_with_warmup",
        "device": device,
    },
)

wandb.define_metric("step")
wandb.define_metric("train/*", step_metric="step")
wandb.define_metric("val/*", step_metric="step")
wandb.define_metric("sys/*", step_metric="step")

config = MyConfig(
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1344,
    max_seq_len=max_seq_len,
    rope_theta=10000.0,
    vocab_size=10000,
)

model = MyTransformer(config).to(device)
model = torch.compile(model)

# NOTE: fused=True only supported on CUDA. Make this conditional.
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, fused=torch.cuda.is_available())

tiny_tokenizer = BPETokenizer.from_files(
    './artifacts/bpe/TinyStoriesV2-GPT4/train-10k-vocab.pkl.gz',
    './artifacts/bpe/TinyStoriesV2-GPT4/train-10k-merges.pkl.gz'
)
train_loader = MyDataLoader('artifacts/bpe/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.memmap',
                            batch_size, max_seq_len, seed=SEED, device=device, reshuffle_on_cycle=False)
val_loader = MyDataLoader('artifacts/bpe/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.memmap',
                          batch_size, max_seq_len, seed=SEED, device=device, reshuffle_on_cycle=False)

torch.set_float32_matmul_precision('high')

wandb.watch(model, log="gradients", log_freq=50)



for step in range(max_steps):

    if step % print_steps == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_total = 0.0
            val_loss_steps = 345
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                val_loss_total += loss.detach().float().item()
            val_loss_avg = val_loss_total / val_loss_steps
            print(f'validaton loss is {val_loss_avg:4f}')
            wandb.log({"step": step, "val/loss": val_loss_avg})

    if step % print_steps == 0 and step != 0:
        start_seq = torch.tensor(tiny_tokenizer.encode("Long ago, in a tiny village"))
        num_samples = 4
        out_seq = generate_seq(model, start_seq, device=device)
        print('-----------------')
        samples = []
        for i in range(num_samples):
            tokens = out_seq[i, :].tolist()
            decoded = tiny_tokenizer.decode(tokens)
            print(decoded)
            print('-----------------')
            samples.append([decoded])

        wandb.log({"step": step, "val/samples": wandb.Table(columns=["text"], data=samples)})

        ckpt_path = f'./artifacts/checkpoints/checkpoint_{step}'
        my_save_checkpoint(model, optimizer, step, ckpt_path)
        artifact = wandb.Artifact("checkpoints", type="model")
        artifact.add_dir('./artifacts/checkpoints', name=f"ckpt_step_{step}")
        wandb.log_artifact(artifact)

    model.train()
    t0 = time.time()
    x, y = train_loader.next_batch()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = my_cosine_learning_rate_schedule(step+1, lr_max=max_lr, lr_min=min_lr, t_w=warmups, t_c=max_steps)
    for g in optimizer.param_groups:
        g['lr' ] = lr
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0) * 1000.0  # millisecs
    tps = (train_loader.B * train_loader.T) / max((t1 - t0), 1e-9)

    print(f"step {step}, loss: {loss.item():.4f}, | lr: {lr:.4e} | norm:{norm:.2f} | dt: {dt:.2f}ms | tps: {tps:.2f}")

    wandb.log({
        "step": step,
        "train/loss": float(loss.item()),
        "train/grad_norm": float(norm),
        "train/lr": float(lr),
        "sys/dt_ms": float(dt),
        "sys/tokens_per_sec": float(tps),
    })

wandb.finish()
