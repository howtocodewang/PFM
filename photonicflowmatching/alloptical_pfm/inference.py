import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
from allopticalPFM.models.dnn import DNN
from initialization import init_params
from allopticalPFM.utils.GShologram import gs_fresnel_batch
from allopticalPFM.utils.preprocess import process_output
import torch.nn.functional as F

class HybridNetworkModel():
    def __init__(self, config):
        super(HybridNetworkModel, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model_pfm = self.get_model()
        self.load()

    def get_model(self):
        model_pfm = DNN(self.device,self.config)
        model_pfm.to(self.device)
        return model_pfm

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

    def valid_step(self, x):
        x = self.holo(x)
        gen_amp, Ein, layer_phase, flows = self.model_pfm(x, flow=True)
        gen_amp_crop = process_output(gen_amp, self.config.output_y_num, self.config.output_x_num, self.config)
        flows_crop = process_output(flows.squeeze().unsqueeze(1), self.config.output_y_num, self.config.output_x_num, self.config)
        flows_crop = flows_crop.squeeze()
        print(flows_crop.shape)
        z = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','0.98', '0.99', '0.995', '0.999', '1.0']
        plt.figure(figsize=(20,5))
        for i in range(flows_crop.shape[0]):
            plt.subplot(1,flows_crop.shape[0], i+1)
            plt.title(f"z = {z[i]} cm", fontsize=8)
            plt.imshow(flows_crop[i].squeeze().detach().cpu().numpy(), cmap='jet')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('./images/checkboard.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

        return gen_amp_crop

    def load(self):
        ## "D:\pythonProject\allopticalPFM\logs\twomoons\model\twomoons.pth"
        ## "D:\pythonProject\allopticalPFM\logs\checkboard\model\checkboard.pth"
        ## "D:\pythonProject\allopticalPFM\logs\riemannsphere\model\Riemannsphere.pth"
        checkpoint = torch.load(r"D:\pythonProject\allopticalPFM\logs\checkboard\model\checkboard.pth")
        self.model_pfm.load_state_dict(checkpoint['model_pfm_state_dict'])


def main():
    torch.backends.cudnn.benchmark = True

    train_config = init_params()
    if train_config.seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = HybridNetworkModel(train_config)

    print('===> Training Start')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input = torch.randn(1, 1, train_config.layer_y_num, train_config.layer_x_num).to(device)
    model.valid_step(input)


if __name__ == '__main__':

    main()