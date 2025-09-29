import torch
import torch.optim as optim
from dataclasses import dataclass
from cs336_basics.layers import MyConfig, MyTransformer
from cs336_basics.data_loader import MyDataLoader
from cs336_basics.utils import my_cross_entropy_loss
import torch.nn.functional as F
import random
import numpy as np

batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"



SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU setups

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = MyConfig(
    num_layers=4, 
    d_model=512, 
    num_heads=16, 
    d_ff=1344, 
    max_seq_len=256,
    rope_theta=10000.0, 
    vocab_size=10000,
)

model = MyTransformer(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

B, T = 4, 32
data_loader = MyDataLoader('artifacts/bpe/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.memmap', B, T, seed=SEED, device=device, reshuffle_on_cycle=False)

for i in range(100):
    x, y = data_loader.next_batch()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    #loss = my_cross_entropy_loss(logits, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item():.4f}")
