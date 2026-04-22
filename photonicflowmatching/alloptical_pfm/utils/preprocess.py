import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
from PIL import Image

def image_preprocess(x, config):
    x = x.type(torch.float)
    b, c, H, W = x.shape
    x = x.view(b, c, H, W)
    if x.shape[-1] != config.layer_x_num:
        x = F.interpolate(x, size=(config.layer_y_num,
                                    config.layer_x_num),
                                    mode='nearest' if config.binary_obj else 'bilinear')
    else:
        x = x

    if config.object_type == 'phase':
        img_phase = x
        img = torch.complex(torch.cos(img_phase), torch.sin(img_phase))
    img_total = F.pad(img, (config.padding, config.padding, config.padding, config.padding))

    return img_total.view(b, c, config.total_y_num, config.total_x_num).squeeze()

def norm(img):
        min_val = torch.amin(img, dim=(-2,-1), keepdim=True)    # [B,C,1,1]
        max_val = torch.amax(img, dim=(-2,-1), keepdim=True)
        img = (img - min_val) / (max_val - min_val)
        return img

def process_output(y, pixelx, pixely, config):
        if y.shape[0] == config.total_y_num:
            y = center_crop(y, [int(config.output_y_num), int(config.output_x_num)]).unsqueeze(0).unsqueeze(0)
        else:
            y = center_crop(y, [int(config.output_y_num), int(config.output_x_num)]).unsqueeze(1)

        if pixelx == config.output_y_num:
             y = y
        else:
             y = F.interpolate(y, size=(pixelx, pixely), mode='area')
        return y

def center_crop_arr(pil_image, image_size):
    # from ADM
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
