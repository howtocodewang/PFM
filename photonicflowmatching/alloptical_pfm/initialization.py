from configobj import ConfigObj
import numpy as np
import numpy.random as random
import scipy.io as sio
import datetime
import argparse
import os

def init_params():
    '''
    Initialize network training parameters.
    tc: training_configurations
    always define x as the prior dimension (matrix rows, along the height),
    y as the following inner (last) dimension
    '''
    # USER-DEFINED GLOBAL PARAMETERS
    # loop train parameters
    parser = argparse.ArgumentParser(description='Supervised Contrastive D2NN')
    parser.add_argument('--ridx_mask', type=float, default=1.7227, help='ridx_mask')
    parser.add_argument('--eff_weight', type=float, default=0.1, help='eff_weight')

    parser.add_argument('--num_layers', type=int, default=9, help='num_layers')  # Number of diffractive layers
    parser.add_argument('--layer_x_num', type=int, default=120,
                        help='layer_x_num')  # Number of neurons of each diffractive layer
    parser.add_argument('--output_x_num', type=int, default=64, help='output_x_num')
    parser.add_argument('--dx0', type=float, default=3.6e-6, help='dx0')

    parser.add_argument('--object_layer_dist', type=float, default=10e-4,
                        help='object_layer_dist')  # Distance between the (encoded) object and first diffractive layer (unit: wavelength)
    parser.add_argument('--layer_layer_dist', type=float, default=10e-4,
                        help='layer_layer_dist')  # Distance between diffractive layers
    parser.add_argument('--layer_sensor_dist', type=float, default=10e-4,
                        help='layer_sensor_dist')  # Layer sensor distances

    parser.add_argument('--nx', type=float, default=0, help='nx')###0
    parser.add_argument('--ny', type=float, default=900, help='ny')###900
    parser.add_argument('--z_vacci', type=float, default=0e-6, help='z_vacci')
    parser.add_argument('--lateral_vacci', type=int, default=1, help='lateral_vacci')
    parser.add_argument('--lr', type=float, default=1e-2, help='lr')

    # parser.add_argument('--image_file', type=str, default='gratings40_a0_13_21_4channel.npy', help='image_file')
    parser.add_argument('--image_file', type=str, default='train_set.npy', help='image_file')
    parser.add_argument('--image_data_path', type=str, default=os.path.join(r'..', 'data'), help='image_data_path')
    parser.add_argument('--output_save_dir', type=str,
                        default=os.path.join(r".\logs",
                                             "20250604-1941-VMath-0.08-0.08-0.08_nLayer700_nOutput448_num3_dx03.6e-06"),
                        help='output_save_dir')
    parser.add_argument('--ckpt_to_load_test', type=str,
                        default=os.path.join('model', 'epoch=005.pth'),
                        help='ckpt_to_load_test')
    parser.add_argument('--test_dir', type=str, default='a0=20', help='test_dir')
    parser.add_argument('--output_save_dir_lin', type=str,
                        default=os.path.join(r"..\logs", "ele_noline_false_after_2"),
                        help='output_save_dir_lin')
    parser.add_argument('--ckpt_to_load_test_lin', type=str,
                        default=os.path.join('model', 'epoch=030.pth'),
                        help='ckpt_to_load_test_lin')
    parser.add_argument('--output_save_dir_cla', type=str,
                        default=os.path.join(r"..\logs", "cla"),
                        help='output_save_dir_cla')
    parser.add_argument('--ckpt_to_load_test_cla', type=str,
                        default=os.path.join('model', 'epoch=020.pth'),
                        help='ckpt_to_load_test_cla')
    opt = parser.parse_args()

    tc = ConfigObj()

    # basic optical parameters
    tc.c = 299792458
    tc.wlength_vc = 635e-9
    tc.freq = tc.c / tc.wlength_vc  # Calculate frequency from wavelength
    tc.ridx_air = 1
    tc.amp_modulation = False  # Consider or not the absorption of the masks
    tc.layer_type = 'phase'  # 'amplitude', 'phase'
    tc.ridx_layer, tc.attenu_factor = opt.ridx_mask, 0  # extract_material_parameter(tc.freq, tc.amp_modulation)
    tc.target_eff = 0.01
    tc.eff_weight = opt.eff_weight

    tc.num_layers = opt.num_layers
    tc.n_viewpoints = 4  # Needs to match dataset

    # Define the distances between the object/display to the first diffractive layer, between diffractive layers, and diffractive layer to sensors
    tc.object_layer_dist, tc.layer_layer_dist, tc.layer_sensor_dist = opt.object_layer_dist, opt.layer_layer_dist, opt.layer_sensor_dist
    tc.z_vacci = opt.z_vacci  # In case we need to add some random misalignment
    tc.psr = 1
    tc.lateral_vacci = opt.lateral_vacci
    # tc.angle = 45 * np.pi / 180 # in radians

    tc.layer_base_thick = 0  # Used only when we consider the absorption from 3D printed layers
    tc.layer_x_num, tc.layer_y_num = opt.layer_x_num, opt.layer_x_num  # number of trainable pixels (excluding the mask boundary)
    tc.dx0, tc.dy0 = opt.dx0, opt.dx0  # grid size of the calculation
    tc.dx, tc.dy = 2 * opt.dx0, 2 * opt.dx0  # pixel size of diffractive neurons

    tc.obj_x_num, tc.obj_y_num = opt.layer_x_num // 2, opt.layer_x_num // 2  # Size of the input image (unit: grid size)
    tc.output_x_num, tc.output_y_num = opt.output_x_num, opt.output_x_num

    tc.theta0 = None
    tc.padding = 0
    tc.layer_size = tc.layer_x_num  # in the number of decoder pixel
    tc.total_x_num = tc.layer_x_num + 2 * tc.padding  # number of pixels of the mask，x是2倍
    tc.total_y_num = tc.layer_y_num + 2 * tc.padding
    # object parameters (unit in meter)
    tc.object_type = 'phase'  # 'amplitude'
    tc.operation = 'sum'  # could also be 'product'

    tc.binary_obj = False

    tc.image_data_path = opt.image_data_path

    # tc.image_data_path = 'test_data'
    tc.image_file = opt.image_file
    tc.image_transform = None  # (list) ['rotate90', 'fliplr', 'flipud','transform']

    # training parameter
    tc.normalize_input = True
    tc.normalize_output = True
    tc.layer_init_method = 'normal'  # layer initilization, 'zero'|'normal'
    tc.batch_size, tc.test_batch_size = 64, 64
    tc.mse_weight, tc.tv_weight = 5, 1

    tc.lr = opt.lr  # 0.5e-2  # 3e-2
    tc.max_epoch = 601
    tc.validation_ratio = 0.15
    tc.seed = None
    tc.checkpoint_save = 5
    tc.checkpoint_print = 1  # tc.old_n, tc.lld_n, tc.lsd_n
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M") + '-VMath-' + str(tc.object_layer_dist) + '-' + str(
        tc.layer_layer_dist) + '-' \
                   + str(tc.layer_sensor_dist) + '_nLayer' + str(tc.layer_y_num) + '_nOutput' + str(
        tc.output_y_num) + '_num' + str(tc.num_layers) + '_dx0' + str(tc.dx0)

    tc.image_save_dir = 'logs/' + datetime_str + '/image'
    tc.model_save_dir = 'logs/' + datetime_str + '/model'
    tc.log_save_dir = 'logs/' + datetime_str + '/tfboard'
    tc.ckpt_to_load = None  # 'logs/20221229-1305-EncMask-15-MaskDet-2_nEnc20_dxEnc400_nDec250_1.7/model/epoch=500.pth' #None

    tc.output_save_dir_lin = opt.output_save_dir_lin
    tc.ckpt_to_load_test_lin = opt.ckpt_to_load_test_lin
    tc.output_save_dir_cla = opt.output_save_dir_cla
    tc.ckpt_to_load_test_cla = opt.ckpt_to_load_test_cla

    tc.output_save_dir = opt.output_save_dir
    tc.ckpt_to_load_test = opt.ckpt_to_load_test
    tc.test_dir = opt.test_dir
    tc.binary = 6
    tc.reduce = 5 * 5
    tc.nx = opt.nx
    tc.ny = opt.ny
    return tc


def extract_material_parameter(frequency, mask_amp_modulation):
    measurement_file = sio.loadmat('../tolga_data/RefIndexMeasurements_1')
    x_freq = measurement_file['f'][0, :]
    y_refidx = measurement_file['n_plastic'][0, :]  # measured refractive index with regard of frequence x
    ridx_mask = np.interp(frequency, x_freq, y_refidx)  # determine the refractive index at the given frequency

    if mask_amp_modulation:
        y_attenu = measurement_file['k_plastic'][0, :]  # measured attenuation factor k with regard of frequence x
        attenu_factor = np.interp(frequency, x_freq,
                                  y_attenu)  # determine the attenuation factor at the given frequency
    else:
        attenu_factor = 0

    return ridx_mask, attenu_factor
