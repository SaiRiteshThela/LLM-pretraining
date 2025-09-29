import os
import random
import numpy as np
import torch

import os
import random
import numpy as np
import torch

class MyDataLoader:
    def __init__(self, source, B=32, T=128, E=1, device=None, seed=None, reshuffle_on_cycle=True, pin_memory=True):
        self.B, self.T = B, T
        self.step = B * T
        self.need = self.step + 1
        self.device = device
        self.reshuffle_on_cycle = reshuffle_on_cycle
        self.pin_memory = pin_memory

        if isinstance(source, (str, os.PathLike)):
            self.data = np.memmap(source, dtype=np.int32, mode='r')
        elif isinstance(source, (np.memmap, np.ndarray)):
            self.data = np.asarray(source, dtype=np.int32).reshape(-1)
        else:
            raise TypeError("source must be a path or a NumPy array/memmap")

        N = len(self.data)
        n_batches = (N - self.need) // self.step + 1
        if n_batches <= 0:
            raise ValueError(f"need at least {self.need} tokens, got {N}")

        base = [i * self.step for i in range(n_batches)]
        self.idxs = base * max(1, int(E))

        self._rng = random.Random(seed)
        self._rng.shuffle(self.idxs)

        self.i = 0

        print(f'loaded {N} tokens')
        print(f'number of batches in one epoch = {(N*E)//(B*T)}')

    def next_batch(self):
        if self.i >= len(self.idxs):
            self.i = 0
            if self.reshuffle_on_cycle:
                self._rng.shuffle(self.idxs)

        s = self.idxs[self.i]
        e = s + self.need
        self.i += 1

        buf = self.data[s:e]
        x = torch.from_numpy(buf[:-1].reshape(self.B, self.T).copy()).long()
        y = torch.from_numpy(buf[1: ].reshape(self.B, self.T).copy()).long()

        if self.pin_memory and self.device is not None and 'cuda' in str(self.device):
            x = x.pin_memory()
            y = y.pin_memory()

        if self.device is not None:
            if 'cuda' in str(self.device):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            else:
                x = x.to(self.device)
                y = y.to(self.device)
        return x, y

    



class MyDataLoaderSmall:
    def __init__(self, source, B=32, T=128, device=None, seed=None, pin_memory=True):
        self.B, self.T = B, T
        self.device = device
        self.pin_memory = pin_memory
        if isinstance(source, (str, os.PathLike)):
            self.data = np.memmap(source, dtype=np.int32, mode='r')
        elif isinstance(source, (np.memmap, np.ndarray)):
            self.data = np.asarray(source, dtype=np.int32).reshape(-1)
        else:
            raise TypeError("source must be a path or a NumPy array/memmap")
        self.N = len(self.data)
        if self.N < self.T + 1:
            raise ValueError(f"need at least {self.T+1} tokens, got {self.N}")
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        max_start = self.N - (self.T + 1)
        starts = self.rng.integers(0, max_start + 1, size=self.B)
        buf = np.stack([self.data[s:s + self.T + 1] for s in starts], axis=0)
        x = torch.from_numpy(buf[:, :-1].copy()).long()
        y = torch.from_numpy(buf[:,  1:].copy()).long()
        if self.pin_memory and self.device is not None and "cuda" in str(self.device):
            x = x.pin_memory(); y = y.pin_memory()
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y

