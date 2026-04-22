import numpy as np
import torch
import torch.nn as nn

class FreeSpaceProp(nn.Module):
    def __init__(self, config, z, rdix_medium):
        super(FreeSpaceProp, self).__init__()
        self.dist = z
        wlengtheff = config.wlength_vc / rdix_medium
        dfx = 1 / (config.total_x_num * config.dx0)
        dfy = 1 / (config.total_y_num * config.dy0)

        fx, fy = torch.meshgrid(
            (torch.arange(config.total_x_num) - config.total_x_num / 2) * dfx,
            (torch.arange(config.total_y_num) - config.total_y_num / 2) * dfy,
        )
        fx = torch.unsqueeze(fx, 0)
        fy = torch.unsqueeze(fy, 0)
        f0 = 1 / wlengtheff

        if config.theta0:
            Q1 = (fx ** 2 + fy ** 2) <= (
                    f0 * torch.sin(torch.tensor(config.theta0 * np.pi / 180))
            ) ** 2
            Q2 = (fx ** 2 + fy ** 2) > (
                    f0 * torch.sin(torch.tensor(config.theta0 * np.pi / 180))
            ) ** 2
        else:
            Q1 = (fx ** 2 + fy ** 2) <= (f0 ** 2)
            Q2 = (fx ** 2 + fy ** 2) > (f0 ** 2)

        prop_window = Q1 * (fx ** 2 + fy ** 2) * (wlengtheff ** 2)
        phase_change_prop = 2 * np.pi * f0 * z * torch.sqrt((1 - prop_window))
        phase_change_cplx_prop = (
                torch.complex(torch.cos(phase_change_prop), torch.sin(phase_change_prop))
                * Q1
        )
        shifted_phase_change_cplx = torch.fft.ifftshift(
            phase_change_cplx_prop, dim=[1, 2]
        )
        shifted_phase_change_cplx = torch.unsqueeze(shifted_phase_change_cplx, dim=0)

        decay_window = Q2 * (fx ** 2 + fy ** 2) * (wlengtheff ** 2)
        amplitude_decay_factor = (
                2 * np.pi * f0 * z * torch.sqrt((decay_window - 1) * Q2) * (-1)
        )
        amplitude_decay = torch.exp(amplitude_decay_factor) * Q2
        amplitude_decay = torch.fft.ifftshift(amplitude_decay, dim=[1, 2])
        amplitude_decay = torch.unsqueeze(amplitude_decay, dim=0)

        self.nearfield_prop = shifted_phase_change_cplx + amplitude_decay

    def forward(self, x):
        ASpectrum = torch.fft.fft2(x)
        ASpectrum_z = torch.mul(self.nearfield_prop.cuda(), ASpectrum)
        output = torch.fft.ifft2(ASpectrum_z)

        return output

    def H(self):
        return self.nearfield_prop