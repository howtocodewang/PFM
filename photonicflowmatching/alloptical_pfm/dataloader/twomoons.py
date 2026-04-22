from sklearn.datasets import make_moons
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def moons_to_amp_batch(B, H, W, n_pts=600, noise=0.06, sigma=0.06, rng=None, device="cpu"):
    """
    Pointclouds of moons -> [-1,1]^2 -> [B,1,H,W] in [0,1]
    """
    if rng is None:
        rng = np.random.RandomState()

    xs = np.linspace(-1, 1, W, dtype=np.float32)
    ys = np.linspace(-1, 1, H, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # [H,W]
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    outs = []
    for _ in range(B):
        pts, _ = make_moons(n_samples=n_pts, noise=noise, random_state=rng.randint(0, 10_000))
        # Normalization [-1,1]
        pts = (pts - pts.mean(0)) / (pts.std(0) + 1e-8)
        pts = pts / 2.5

        den = np.zeros((H, W), dtype=np.float32)

        for (px, py) in pts:
            dx = XX - px; dy = YY - py
            den += np.exp(-(dx*dx + dy*dy) * inv2s2)

        den -= den.min()
        mx = den.max()
        if mx > 1e-8:
            den /= mx
        outs.append(den[None, None, ...])  # [1,1,H,W]

    amp = torch.from_numpy(np.concatenate(outs, axis=0)).to(device)  # [B,1,H,W] in [0,1]
    return amp

class MoonsRasterDataset(Dataset):
    def __init__(self, length, H, W, device=device, n_pts=600, noise=0.06, sigma=0.06, seed=0):
        self.len = length
        self.H, self.W = H, W
        self.device = device
        self.n_pts = n_pts
        self.noise = noise
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

    def __len__(self): return self.len

    def __getitem__(self, idx):
        amp = moons_to_amp_batch(
            B=1, H=self.H, W=self.W,
            n_pts=self.n_pts, noise=self.noise, sigma=self.sigma,
            rng=self.rng, device=self.device
        )  # [1,1,H,W]
        return amp[0]  # [1,H,W]


def make_twomoons_loader(batch_size,H,W,seed):
    ds = MoonsRasterDataset(length=20_000, H=H, W=W, device=device,
                            n_pts=600, noise=0.06, sigma=0.06, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0)