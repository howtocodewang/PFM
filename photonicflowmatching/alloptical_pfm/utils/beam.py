import torch
import math

def gaussian_beam(shape,
                  w0=1.0,
                  z=0.0,
                  wavelength=0.635,
                  z0=0.0,
                  device="cpu"):
    """
    Generate the Gaussian beam (TEM00)

    parameters:
        shape: (H, W)
        w0:   waist radius
        z:    distance
        z0:   waist position
        device: "cpu" or "cuda"

    return:
        E: [H, W] complex amplitude (torch.complex64)
    """

    H, W = shape if isinstance(shape, tuple) else (shape, shape)
    x = torch.linspace(-5 * w0, 5 * w0, W, device=device)
    y = torch.linspace(-5 * w0, 5 * w0, H, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # to tensor
    z = torch.tensor(z, dtype=torch.float32, device=device)
    z0 = torch.tensor(z0, dtype=torch.float32, device=device)
    wavelength = torch.tensor(wavelength, dtype=torch.float32, device=device)

    # wave number
    k = 2 * math.pi / wavelength

    # Rayleigh length
    zR = math.pi * w0 ** 2 / wavelength

    # spot size
    wz = w0 * torch.sqrt(1.0 + ((z - z0) / zR) ** 2)

    # radius
    Rz = torch.where(
        (z - z0) == 0,
        torch.tensor(float("inf"), device=device),
        (z - z0) * (1 + (zR / (z - z0)) ** 2)
    )

    # Gouy phase
    gouy = torch.atan((z - z0) / zR)

    # r^2
    r2 = X ** 2 + Y ** 2

    # amplitude distribution
    amp = (w0 / wz) * torch.exp(-r2 / wz ** 2)

    # phase distribution
    phase = torch.exp(-1j * (k * (z - z0) + k * r2 / (2 * Rz) - gouy))

    return (amp * phase).to(torch.complex64)