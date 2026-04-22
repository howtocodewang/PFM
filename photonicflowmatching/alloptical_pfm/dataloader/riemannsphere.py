import math, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def _rand_so3(device, dtype):
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / (axis.norm() + 1e-8)
    angle = torch.rand(1, device=device, dtype=dtype) * 2*math.pi
    x,y,z = axis
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1-c
    R = torch.stack([
        torch.stack([c + x*x*C,    x*y*C - z*s, x*z*C + y*s], dim=0),
        torch.stack([y*x*C + z*s,  c + y*y*C,   y*z*C - x*s], dim=0),
        torch.stack([z*x*C - y*s,  z*y*C + x*s, c + z*z*C  ], dim=0),
    ], dim=0)  # [3,3]
    return R

class RiemannSphereCheckerboard(Dataset):
    """
    Generate the Riemann sphere
    """
    def __init__(self, num=10000, H=64, W=64,
                 n_phi_range=(6, 10),
                 n_theta_range=(4, 8),
                 rotate=True,
                 edge_softness_px=0.8,
                 blur_sigma=0.0,
                 contrast=(0.0, 1.0),
                 seed=42,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        self.num, self.H, self.W = num, H, W
        self.n_phi_range = n_phi_range
        self.n_theta_range = n_theta_range
        self.rotate = rotate
        self.edge_softness_px = edge_softness_px
        self.blur_sigma = blur_sigma
        self.contrast = contrast
        self.device, self.dtype = device, dtype
        random.seed(seed); torch.manual_seed(seed)

        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')   # [H,W]
        self.X = X; self.Y = Y
        self.R2 = X**2 + Y**2
        self.mask_disk = (self.R2 <= 1.0).float()

        if edge_softness_px and edge_softness_px > 0:
            r = torch.sqrt(self.R2 + 1e-8)
            k = 4.0 / max(edge_softness_px, 1e-6)
            self.soft_edge = torch.sigmoid(-k * (r - 1.0))
        else:
            self.soft_edge = self.mask_disk

        if self.blur_sigma and self.blur_sigma > 0:
            rad = 2
            xs = torch.arange(-rad, rad+1, device=device, dtype=dtype)
            k1 = torch.exp(-0.5 * (xs / self.blur_sigma)**2); k1 /= k1.sum()
            self.k2d = torch.outer(k1, k1).view(1,1,2*rad+1,2*rad+1)
        else:
            self.k2d = None

    def __len__(self): return self.num

    def __getitem__(self, idx):
        X, Y, R2 = self.X, self.Y, self.R2
        device, dtype = self.device, self.dtype

        Z = torch.sqrt(torch.clamp(1.0 - R2, min=0.0))                  # [H,W]
        P = torch.stack([X, Y, Z], dim=-1)                              # [H,W,3]

        if self.rotate:
            R = _rand_so3(device, dtype)                                # [3,3]
            Pr = (P @ R.T)                                              # [H,W,3]
        else:
            Pr = P

        Xr, Yr, Zr = Pr[..., 0], Pr[..., 1], Pr[..., 2]

        theta = torch.acos(torch.clamp(Zr, -1.0, 1.0))                  # [0, π]
        phi   = torch.atan2(Yr, Xr)                                     # (-π, π]

        n_phi   = random.randint(*self.n_phi_range)
        n_theta = random.randint(*self.n_theta_range)

        phi_u   = (phi + math.pi) / (2*math.pi)                         # [0,1)
        theta_u = theta / math.pi                                       # [0,1)
        i = torch.floor(phi_u   * n_phi).to(torch.int32)
        j = torch.floor(theta_u * n_theta).to(torch.int32)
        checker = ((i + j) % 2).float()                                 # 0/1

        lo, hi = self.contrast
        tex = lo + (hi - lo) * checker                                  # [H,W]

        amp = tex * self.soft_edge                                      # [H,W]
        amp = amp * self.mask_disk

        if self.k2d is not None:
            amp = F.conv2d(amp.view(1,1,self.H,self.W), self.k2d, padding=self.k2d.shape[-1]//2).view(self.H,self.W)

        return amp.clamp(0,1).unsqueeze(0)                              # [1,H,W]

def make_riemannsphere_loader(batch_size=32, size=64, n_samples=10000,
                               n_phi_range=(6,10), n_theta_range=(4,8),
                               rotate=True, edge_softness_px=0.8,
                               blur_sigma=0.0, contrast=(0.0,1.0),
                               shuffle=True, device="cpu"):
    ds = RiemannSphereCheckerboard(num=n_samples, H=size, W=size,
                                   n_phi_range=n_phi_range, n_theta_range=n_theta_range,
                                   rotate=rotate, edge_softness_px=edge_softness_px,
                                   blur_sigma=blur_sigma, contrast=contrast,
                                   device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    loader = make_riemannsphere_loader(batch_size=8, size=128,
                                        n_samples=10000,
                                        n_phi_range=(6,10), n_theta_range=(4,8),
                                        rotate=True, edge_softness_px=1.2,
                                        blur_sigma=0.6, contrast=(0.0,1.0))
    batch = next(iter(loader))            # [B,1,H,W]
    grid = torch.cat([batch[:8]], dim=0)

    fig, axes = plt.subplots(2,4, figsize=(7,3.6))
    for ax, img in zip(axes.flatten(), grid):
        ax.imshow(img[0].numpy(), vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
