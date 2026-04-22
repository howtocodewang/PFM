import math, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CheckerboardDataset(Dataset):
    """
    Generate the checkboard dataset
    """
    def __init__(self, num=10000, H=64, W=64,
                 squares=(6, 10),
                 translate=0.0,
                 rotate_deg=0.0,
                 blur_sigma=0.0,
                 invert_p=0.0,
                 seed=42):
        super().__init__()
        self.num, self.H, self.W = num, H, W
        self.squares = squares
        self.translate = translate
        self.rotate_deg = rotate_deg
        self.blur_sigma = blur_sigma
        self.invert_p = invert_p
        random.seed(seed)

        y = torch.arange(H, dtype=torch.float32) / H
        x = torch.arange(W, dtype=torch.float32) / W
        self.Y, self.X = torch.meshgrid(y, x, indexing='ij')  # [H,W]

    def __len__(self): return self.num

    def _gauss_kernel(self, sigma):
        rad = 2
        xs = torch.arange(-rad, rad+1, dtype=torch.float32)
        k1 = torch.exp(-0.5 * (xs / (sigma + 1e-8))**2)
        k1 = k1 / k1.sum()
        k2d = torch.outer(k1, k1)
        k2d = (k2d / k2d.sum()).view(1,1,2*rad+1,2*rad+1)
        return k2d

    def __getitem__(self, idx):
        H, W = self.H, self.W
        X, Y = self.X, self.Y

        n = random.randint(self.squares[0], self.squares[1])
        dx = (random.uniform(-self.translate, self.translate)) % 1.0
        dy = (random.uniform(-self.translate, self.translate)) % 1.0
        Xs = (X + dx) % 1.0
        Ys = (Y + dy) % 1.0

        board = ((torch.floor(Xs * n) + torch.floor(Ys * n)) % 2).float()  # [H,W], 0/1
        img = board.unsqueeze(0)  # [1,H,W]

        if self.rotate_deg > 0:
            ang = random.uniform(-self.rotate_deg, self.rotate_deg) * math.pi / 180.0
            cos, sin = math.cos(ang), math.sin(ang)
            theta = torch.tensor([[cos, -sin, 0.0],
                                  [sin,  cos, 0.0]], dtype=torch.float32).unsqueeze(0)  # [1,2,3]
            grid = F.affine_grid(theta, size=(1,1,H,W), align_corners=False)
            img = F.grid_sample(img.unsqueeze(0), grid, mode='nearest',
                                padding_mode='zeros', align_corners=False).squeeze(0)  # [1,H,W]

        if self.blur_sigma and self.blur_sigma > 0:
            k = self._gauss_kernel(self.blur_sigma)
            img = F.conv2d(img.unsqueeze(0), k, padding=2).squeeze(0)

        if self.invert_p > 0 and random.random() < self.invert_p:
            img = 1.0 - img

        return img.clamp(0, 1)  # [1,H,W], float32


def make_checkerboard_loader(batch_size=32, size=64, n_samples=10000,
                             squares=(6,10), translate=0.15, rotate_deg=10.0,
                             blur_sigma=0.0, invert_p=0.0, shuffle=True):
    ds = CheckerboardDataset(num=n_samples, H=size, W=size,
                             squares=squares,
                             translate=translate,
                             rotate_deg=rotate_deg,
                             blur_sigma=blur_sigma,
                             invert_p=invert_p)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=True)
