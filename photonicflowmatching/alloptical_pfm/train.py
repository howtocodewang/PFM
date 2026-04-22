import os
import random
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.parallel
from allopticalPFM.models.dnn import DNN
from initialization import init_params
import torch.optim as optim
from pytorch_ssim import SSIM
from loss_functions import DiffractionEfficiency
from allopticalPFM.utils.GShologram import gs_fresnel_batch
from allopticalPFM.dataloader.checkerboard import make_checkerboard_loader
from allopticalPFM.dataloader.riemannsphere import make_riemannsphere_loader
from allopticalPFM.dataloader.twomoons import make_twomoons_loader
from allopticalPFM.utils.preprocess import process_output
import torch.nn.functional as F

class HybridNetworkModel():
    def __init__(self, config):
        super(HybridNetworkModel, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.loss_func = self.init_loss_func()
        self.model_pfm = self.get_model()
        self.optimizer_pfm = self.init_optimizer()
        #self.load()
        self.is_training = True
        self.SSIM = SSIM()

    def get_model(self):
        model_pfm = DNN(self.device,self.config)
        model_pfm.to(self.device)
        return model_pfm

    def init_optimizer(self):
        optimizer_pfm = optim.Adam(
            self.model_pfm.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),weight_decay=1e-6)
        return optimizer_pfm

    def init_loss_func(self):
        loss_func = {}
        loss_func['mse'] = nn.MSELoss(reduction='mean')
        loss_func['diff_eff'] = DiffractionEfficiency()
        loss_func['ssim'] = SSIM()

        return loss_func

    def calculate_loss(self,x0, x, y):
        losses = {}
        x_norm = torch.div(
                    x, torch.amax(x, dim=[-2, -1], keepdim=True))
        losses['ssim'] =  self.loss_func['ssim'](x_norm**2, y)
        losses['mse'] = self.loss_func['mse'](x_norm**2, y)
        losses['diff_eff'] = self.loss_func['diff_eff'](x,torch.abs(x0))

        # if losses['diff_eff'] < 0.65:
        #     loss = 2 + 5 * losses['mse'] - losses['ssim'] - losses['diff_eff']
        # else:
        loss = 1 + 5 * losses['mse'] - losses['ssim']

        return loss,losses['mse'],losses['ssim'],losses['diff_eff']

    def holo(self,x):
        x = F.interpolate(x, size = (self.config.output_y_num, self.config.output_x_num), mode='bilinear')
        padx = (self.config.total_x_num - self.config.output_x_num)//2
        pady = (self.config.total_y_num - self.config.output_y_num)//2
        x = F.pad(x, (pady, pady, padx, padx))
        out = gs_fresnel_batch(
            target_amp=x,
            z=self.config.num_layers*self.config.object_layer_dist,
            iters=1, phase_only=True
        )

        return out["holo_phase"]

    def train_step(self, x,tgt, j):
        self.optimizer_pfm.zero_grad()
        x = self.holo(x)
        gen_amp, Ein, layer_phase = self.model_pfm(x,j)
        gen_amp_crop = process_output(gen_amp, self.config.output_y_num, self.config.output_x_num, self.config)
        loss,mse,ssim,diff = self.calculate_loss(Ein,gen_amp_crop, tgt)
        loss.backward()
        self.optimizer_pfm.step()
        return loss,gen_amp_crop,mse,ssim,diff

    def load(self):
        checkpoint = torch.load(r"D:\pythonProject\CLIP_D2NN\logs\20251021-2149-VMath-8e-09-8e-09-8e-09_nLayer120_nOutput64_num5_dx04e-07\model\epoch=1100.pth")
        self.model_pfm.load_state_dict(checkpoint['model_dnn_state_dict'])
        self.optimizer_pfm.load_state_dict(checkpoint['optimizer_dnn_state_dict'])

    def save(self):
        torch.save({
            'epoch': self.epoch,
            'model_pfm_state_dict': self.model_pfm.state_dict(),
            'optimizer_pfm_state_dict': self.optimizer_pfm.state_dict(),
        },
            self.config.model_save_dir + '/epoch=%03d.pth' % self.epoch)


def main():
    torch.backends.cudnn.benchmark = True

    train_config = init_params()
    if not os.path.exists(train_config.image_save_dir):
        os.makedirs(train_config.image_save_dir)
    if not os.path.exists(train_config.model_save_dir):
        os.makedirs(train_config.model_save_dir)
    if not os.path.exists(train_config.log_save_dir):
        os.makedirs(train_config.log_save_dir)
    writer = SummaryWriter(train_config.log_save_dir)
    if train_config.seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batchsize = 1

    model = HybridNetworkModel(train_config)

    #train_loader = make_twomoons_loader(batchsize,H=64,W=64,seed=seed)
    # train_loader = make_riemannsphere_loader(batch_size=batchsize, size=64,
    #                                     n_samples=32,
    #                                     n_phi_range=(6, 10), n_theta_range=(4, 8),
    #                                     rotate=True, edge_softness_px=1.2,
    #                                     blur_sigma=0.6, contrast=(0.0, 1.0))

    train_loader = make_checkerboard_loader(
        batch_size=batchsize,
        size=64,
        n_samples=20000,
        squares=(4, 4),  # 每边 6~10 个小格，随机构成棋盘
        translate=0.0,  # 随机平移（相对坐标）
        rotate_deg=0,  # 轻微随机旋转
        blur_sigma=0.0,  # 如需软边缘可设 0.5
        invert_p=0.0,  # 一半样本黑白反转（可视化更丰富）
        shuffle=True
    )

    print('===> Training Start')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt = next(iter(train_loader))
    tgt = F.interpolate(tgt, size=(train_config.output_y_num, train_config.output_x_num), mode='area')

    # training
    for epoch in range(0, 10001, 1):
        input = torch.randn(1, 1, train_config.layer_y_num, train_config.layer_x_num).to(device)
        dnn_loss, gen_map,mse,ssim,diff = model.train_step(input,tgt.to(device), epoch)
        train_loss = dnn_loss.item()
        train_mse = mse.item()
        train_ssim = ssim.item()
        train_diff = diff.item()

        model.epoch = epoch
        print('<epoch:{:3d}, loss:{:.3e}> '
              .format(epoch, train_loss))
        writer.add_scalars('Loss', {'Train': float(train_loss)}, epoch)
        writer.add_scalars('MSE', {'Train': float(train_mse)}, epoch)
        writer.add_scalars('SSIM', {'Train': float(train_ssim)}, epoch)
        writer.add_scalars('Diff', {'Train': float(train_diff)}, epoch)

        if epoch % 100 == 0:
            gen_map = gen_map**2
            vutils.save_image(
                gen_map.detach(),
                '%s/train_epoch_%03d.png'
                % (train_config.image_save_dir, epoch),
                normalize=True, value_range=(float(gen_map.min().item()), float(gen_map.max().item())))
            vutils.save_image(
                tgt.detach(),
                '%s/train_tgt_epoch_%03d.png'
                % (train_config.image_save_dir, epoch),
                normalize=True, value_range=(0, 1))
        if epoch % 100 == 0:
            print('Saving the model.')
            model.save()

if __name__ == '__main__':

    main()
