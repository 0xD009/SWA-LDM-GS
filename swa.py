import torch
import numpy as np
from functools import reduce

def binlist2int(binlist):
    return reduce(lambda x, y: x * 2 + y, binlist)

def sequential_map(x):
    i = x // (64 * 64)
    j = (x // 64) % 64
    k = x % 64
    return i, j, k

def key_channel_enhance(M, R):
    key = [0] * M
    key_channel = torch.randn(1, 64, 64)
    for r in range(R):
        for m in range(M):
            i, j, k = sequential_map(r * M + m)
            km_r = 1 if key_channel[i, j, k] > 0 else 0
            if r == 0:
                key[m] = km_r
            elif km_r != key[m]:
                p = m + 1
                while True:
                    i2, j2, k2 = sequential_map(r * M + p)
                    new_bit = 1 if key_channel[i2, j2, k2] > 0 else 0
                    if new_bit == key[m]:
                        temp = key_channel[i, j, k].item()  
                        key_channel[i, j, k] = key_channel[i2, j2, k2].item()
                        key_channel[i2, j2, k2] = temp
                        break
                    p += 1
    return key_channel, binlist2int(key)

def exact_k(key_channel, M, R):
    codeword = (key_channel > 0).float().flatten().reshape(-1, M)[:R, :].mean(dim=0)
    k = (codeword > 0.5).int().tolist()
    return binlist2int(k)

def FY_shuffle(z, k):
    z_flatten = z.flatten().clone()
    generator = np.random.Generator(np.random.PCG64(seed=k))
    for i in range(z_flatten.shape[0]-1, 0, -1):
        j = generator.integers(0, i)
        temp = z_flatten[i].item()
        z_flatten[i] = z_flatten[j].item()
        z_flatten[j] = temp
    return z_flatten.reshape(z.shape)

def FY_inverse_shuffle(z, k):
    dummy = torch.arange(z.numel())
    generator = np.random.Generator(np.random.PCG64(seed=k))
    for i in range(z.numel()-1, 0, -1):
        j = generator.integers(0, i)
        temp = dummy[i].item()
        dummy[i] = dummy[j].item()
        dummy[j] = temp
    original_idx = torch.argsort(dummy)
    return z.flatten()[original_idx].reshape(z.shape)

