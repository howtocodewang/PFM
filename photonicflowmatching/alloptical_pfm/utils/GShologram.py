import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复（可能不稳定/有隐患）
os.environ["OMP_NUM_THREADS"] = "1"          # 可选：限制线程

import torch
from allopticalPFM.utils.beam import gaussian_beam
from allopticalPFM.initialization import init_params
from allopticalPFM.utils.propagation import FreeSpaceProp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    denom = x.amax(dim=(-2, -1), keepdim=True) + eps
    return x / denom

# ---------- 批量 GS（菲涅尔/角谱版） ----------
def gs_fresnel_batch(
    target_amp: torch.Tensor,  # 形状 (B, C, M, 2M)，float，需要在 device 上
    z: float,
    iters: int = 10,
    phase_only: bool = True
):
    """
    return：
      - 'holo_phase': (B,C,H,W) hologram
      - 'recon_amp' : (B,C,H,W) reconstruction amplitude
      - 'rms'       : (B,iters)
    """
    assert target_amp.ndim == 4, "target_amp must be (B,C,H,W)"
    device = target_amp.device
    B, C, H, W = target_amp.shape
    eps = 1e-12

    # normalization
    tgt = normalize(target_amp)
    config = init_params()
    u = gaussian_beam(H,w0=1e-6,z=1e-5,wavelength=config.wlength_vc,device=device).unsqueeze(0).unsqueeze(0)

    rms = torch.zeros((B, iters), device=device, dtype=torch.float32)

    for it in range(iters):
        # forward propagation
        uz = FreeSpaceProp(config,z,config.ridx_air)(u)

        # change the amplitude
        uz_phase = torch.exp(1j * torch.angle(uz))      # (B,C,H,W), complex
        uz = tgt.to(uz.dtype) * uz_phase                # real*complex -> complex

        # calculate rms
        uz_amp = torch.abs(uz)
        uz_amp_norm = uz_amp / (uz_amp.amax(dim=(-2, -1), keepdim=True) + eps)
        err = tgt - uz_amp_norm
        rms[:, it] = torch.sqrt(torch.mean(err**2, dim=(1, 2, 3)))

        # back propagation
        u = FreeSpaceProp(config,-z,config.ridx_air)(uz)

        # SLM simulation
        if phase_only:
            u = torch.exp(1j * torch.angle(u))          # amplitude = I (matrix)
        else:
            amp = torch.clamp(torch.abs(u), 0.0, 1.0)
            u = amp * torch.exp(1j * torch.angle(u))

    # hologram
    holo_phase = torch.angle(u)                          # float in [-pi,pi]

    # reconstruction light
    uz_final = FreeSpaceProp(config,z,config.ridx_air)(u)
    recon_amp = torch.abs(uz_final)**2
    recon_amp = recon_amp / (recon_amp.amax(dim=(-2, -1), keepdim=True) + eps)

    return {
        "holo_phase": holo_phase,        # (B,C,H,W), float
        "recon_amp": recon_amp,          # (B,C,H,W), float in [0,1]
        "rms": rms                       # (B,iters), float
    }


if __name__ =="__main__":
    # test the GS algorithmn
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((700,700))])
    dataset = datasets.MNIST(root="your dir", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    imgs = next(iter(data_loader))[0].cuda()
    out = gs_fresnel_batch(
        target_amp=imgs,z=0.08,
        iters=100, phase_only=True
    )

    holo_phase_batch = out["holo_phase"] + torch.pi
    recon_amp_batch = out["recon_amp"]
    plt.subplot(1, 2, 1)
    plt.imshow(holo_phase_batch[0].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(recon_amp_batch[0].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.show()