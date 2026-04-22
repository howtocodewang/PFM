import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allopticalPFM.utils.propagation import FreeSpaceProp
from allopticalPFM.plot import save_images
from allopticalPFM.utils.beam import gaussian_beam

class DNN(nn.Module):
    def __init__(self, device, config, wlength_vc=635e-9, FreeSpaceProp_SLM_test=None, gen_type='twomoons'):
        super().__init__()
        self.config = config
        layerBlock_phase = NetworkBlockPhase_nonlinear
        self.D2NN_p = nn.ModuleList()
        self.device = device

        for _ in range(config.num_layers):
            self.D2NN_p.append(layerBlock_phase(self.config))

        if FreeSpaceProp_SLM_test is None:
            self.H = FreeSpaceProp(self.config,
                                   self.config.object_layer_dist,
                                   self.config.ridx_air).H()
        else:
            self.H = FreeSpaceProp_SLM_test(self.config,
                                            self.config.object_layer_dist,
                                            self.config.ridx_air).H()
        self.wlength_vc = wlength_vc
        self.gen_type = gen_type

    def prop(self, x):
        ASpectrum = torch.fft.fft2(x)
        ASpectrum_z = torch.mul(self.H.to(x.device), ASpectrum)
        output = torch.fft.ifft2(ASpectrum_z)
        return output

    def forward(self, x, save_figs=False, rgb='r', flow='False'):
        if save_figs is True:
            save_images(x, fr"./images/{self.gen_type}/{rgb}/phase",
                        dir2=f"phase_dynamic{0}.png",
                        cmap="RdPu")

        Ein = gaussian_beam(x.shape[-2], w0=1e-6, z=1e-5,
                            wavelength=self.config.wlength_vc,device=self.device) * torch.exp(1j * x)
        Ein2 = Ein
        E_cplx = Ein2
        nonlinear = E_cplx
        layer_phases = []
        flows = []
        if flow is True:
            flows.append(E_cplx)
        for i in range(0, len(self.D2NN_p)):
            blocks_p = self.D2NN_p[i]
            if flow is True:
                flows.append(torch.fft.fftshift((FreeSpaceProp(self.config, self.config.object_layer_dist / 2,
                                                                   self.config.ridx_air)(E_cplx)), dim=-2))
            E_cplx = self.prop(E_cplx)
            E_cplx, layer_phase, layer_cplx = blocks_p(E_cplx, nonlinear, i + 1, save_figs, self.gen_type)
            nonlinear = layer_cplx
            layer_phases.append(layer_phase)
            if flow is True:
                flows.append(E_cplx)

        if flow is True:
            flows.append(torch.fft.fftshift((FreeSpaceProp(self.config, self.config.object_layer_dist / 2,
                                                               self.config.ridx_air)(E_cplx)), dim=-2))

        E_cplx = self.prop(E_cplx)
        output = torch.abs(E_cplx).squeeze()
        if flow is True:
            flows.append(E_cplx)

        if flow is True:
            return output, Ein, layer_phases, torch.stack(flows, dim=1)
        else:
            return output, Ein, layer_phases


class NetworkBlockPhase_nonlinear(nn.Module):
    def __init__(self, config, z=0, rdix_medium=1, in_channnels=1):
        super(NetworkBlockPhase_nonlinear, self).__init__()
        self.register_parameter(
            "layer_phase",
            nn.Parameter(
                torch.Tensor(1, in_channnels, config.layer_y_num // 2, config.layer_x_num // 2),
                requires_grad=True,
            ),
        )
        self.register_parameter(
            "layer_phase2",
            nn.Parameter(
                torch.Tensor(1, in_channnels, config.layer_y_num // 2, config.layer_x_num // 2),
                requires_grad=True,
            ),
        )
        self.config = config
        if config.layer_init_method == "zero":
            nn.init.zeros_(self.layer_phase.data)
            nn.init.zeros_(self.layer_phase2.data)
        elif config.layer_init_method == "normal":
            nn.init.normal_(self.layer_phase.data, mean=0, std=0.5)
            nn.init.normal_(self.layer_phase2.data, mean=0, std=0.5)

        self.pad_x = config.total_x_num // 2 - config.layer_x_num // 2
        self.pad_y = config.total_y_num // 2 - config.layer_y_num // 2

        self.phase_change_factor = (
                2 * np.pi * (config.ridx_layer - 1) * config.freq / config.c
        )
        self.amp_decay_factor = (
                2 * np.pi * config.attenu_factor * config.freq / config.c
        )

        self.layer_base_thick = config.layer_base_thick
        self.thickness = z
        self.n = rdix_medium
        self.wavelength = config.wlength_vc

    def forward(self, x, img, i=None, save_fig=False, gen_type='mnist', rgb='r'):
        device = x.device
        img_phase = torch.angle(img)[..., self.pad_y: self.pad_y + self.config.layer_y_num,
                    self.pad_x: self.pad_x + self.config.layer_x_num]
        img_phase = F.interpolate(img_phase, size=(self.config.layer_y_num // 2, self.config.layer_x_num // 2),
                                  mode='bilinear', align_corners=False)

        b = torch.sigmoid(self.layer_phase) * 2 * np.pi
        dynamic = self.layer_phase2 * img_phase
        layer_phase = b + dynamic
        layer_phase2 = layer_phase

        layer_phase = torch.remainder(layer_phase, 2 * np.pi)

        if save_fig is True:
            self.save_phase(i, layer_phase, layer_phase2, b, dynamic, img_phase, gen_type, rgb)

        layer_phase = F.interpolate(
            layer_phase,
            size=(self.config.layer_y_num, self.config.layer_x_num),
            mode="nearest",
        )

        layer_cplx = torch.complex(torch.cos(layer_phase), torch.sin(layer_phase))

        layer_cplx = F.pad(layer_cplx, (self.pad_x, self.pad_x, self.pad_y, self.pad_y))
        layer_cplx = layer_cplx.to(device)
        x = x.to(device)

        output = torch.mul(layer_cplx, x)
        return output, layer_phase, layer_cplx

    def save_phase(self, i, layer_phase, layer_phase2, b, dynamic, img_phase, gen_type='mnist', rgb='r'):
        phase_b = torch.sigmoid(self.layer_phase.cuda())
        phase_w = self.layer_phase2.cuda()
        phase_w = (phase_w - phase_w.amin(dim=[-2, -1], keepdim=True)) / (
                phase_w.amax(dim=[-2, -1], keepdim=True) - phase_w.amin(dim=[-2, -1], keepdim=True))
        residual_fixed = layer_phase2 - b
        residual_dynamic = layer_phase2 - dynamic
        residual_previous = layer_phase2 - img_phase
        save_images(layer_phase, fr"./images/{gen_type}/{rgb}/phase",
                    f"phase_dynamic{i}.png", cmap="RdPu", value_range=(0, 2 * np.pi))
        save_images(phase_w, fr"./images/{gen_type}/{rgb}/phase",
                    f"phase_w{i}.png", cmap="RdPu", value_range=(0, 1))
        save_images(phase_b, fr"./images/{gen_type}/{rgb}/phase",
                    f"phase_b{i}.png", cmap="RdPu", value_range=(0, 1))
        save_images(layer_phase2[0], fr"./images/{gen_type}/{rgb}/phase",
                    f"phase_dynamic_total{i}.png", cmap="RdPu", colorbar=True)
        save_images(residual_fixed[0], fr"./images/{gen_type}/{rgb}/phase",
                    f"residual_fixed{i}.png", cmap="bwr", colorbar=True)
        save_images(residual_dynamic[0], fr"./images/{gen_type}/{rgb}/phase",
                    f"residual_dynamic{i}.png", cmap="bwr", colorbar=True)
        save_images(residual_previous[0], fr"./images/{gen_type}/{rgb}/phase",
                    f"residual_previous{i}.png", cmap="bwr", colorbar=True)
        save_images(b, fr"./images/{gen_type}/{rgb}/phase", f"fixed_b{i}.png",
                    cmap="RdPu", colorbar=True)
        save_images(self.layer_phase2, fr"./images/{gen_type}/{rgb}/phase",
                    f"fixed_w{i}.png", cmap="RdPu", colorbar=True)
        save_images(dynamic[0], fr"./images/{gen_type}/{rgb}/phase",
                    f"dynamic{i}.png", cmap="RdPu", colorbar=True)