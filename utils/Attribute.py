import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def check_image_size(x, padder_size=256):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    return x


def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = 2 * normalized_data - 1
    return normalized_data


def normalize_data_torch(data):
    """Normalize the data to the range [0, 1]."""
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def calculate_spatially_varying_exposure_HSI(image_path, base_exposure=0.55, adjustment_amplitude=0.15):
    img = Image.open(image_path).convert('RGB')
    rgb_img = np.array(img).astype(np.float32) / 255.0

    l_channel = np.mean(rgb_img, axis=2)
    l_avg = np.mean(l_channel)

    norm_diff = normalize_data(l_avg - l_channel)

    exposure_map = base_exposure + adjustment_amplitude * norm_diff
    exposure_map = exposure_map.astype(np.float32)[:, :, np.newaxis]
    exposure_map = torch.tensor(exposure_map).permute(2, 0, 1).unsqueeze(0)

    if torch.cuda.is_available():
        exposure_map = exposure_map.cuda()

    return exposure_map


def calculate_color_map(input, Retinex):
    L, color_map = Retinex(input)
    return color_map


def calculate_color_map_fix(input, Retinex):
    input = torch.pow(input, 0.25)
    data_low = input.squeeze(0) / 20

    data_max_r = data_low[0].max()
    data_max_g = data_low[1].max()
    data_max_b = data_low[2].max()
    
    color_max = torch.zeros((data_low.shape[0], data_low.shape[1], data_low.shape[2])).cuda()
    color_max[0, :, :] = data_max_r * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    color_max[1, :, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    color_max[2, :, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    
    data_color = data_low / (color_max + 1e-6)
    return data_color.unsqueeze(0)


class L_structure(nn.Module):
    def __init__(self):
        super(L_structure, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        
        return D_left + D_right + D_up + D_down


class L_exp2(nn.Module):
    def __init__(self, patch_size):
        super(L_exp2, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, y):
        x = torch.mean(x, 1, keepdim=True)
        mean_x = self.pool(x)
        mean_y = self.pool(y)
        d = torch.mean(torch.pow(mean_x - mean_y, 2))
        return d

class L_structure2(nn.Module):
    def __init__(self):
        super(L_structure2, self).__init__()

    def forward(self, input, target):
        H,W = input.shape[-2:]
        x_fft = torch.fft.rfft2(input+1e-8, norm='backward')
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        real_uni = 1 * torch.cos(x_pha)+1e-8
        imag_uni = 1 * torch.sin(x_pha)+1e-8
        x_uni = torch.complex(real_uni, imag_uni)+1e-8
        x_uni = torch.abs(torch.fft.irfft2(x_uni, s=(H, W), norm='backward'))
        x_g = torch.gradient(x_uni,axis=(2,3),edge_order=2)
        x_g_x  = x_g[0];x_g_y = x_g[1]
        
        y_fft = torch.fft.rfft2(target+1e-8, norm='backward')
        y_amp = torch.abs(y_fft)
        y_pha = torch.angle(y_fft)
        real_uni = 1 * torch.cos(y_pha)+1e-8
        imag_uni = 1 * torch.sin(y_pha)+1e-8
        y_uni = torch.complex(real_uni, imag_uni)+1e-8
        y_uni = torch.abs(torch.fft.irfft2(y_uni, s=(H, W), norm='backward'))
        y_g = torch.gradient(y_uni,axis=(2,3),edge_order=2)
        y_g_x  = y_g[0];y_g_y =y_g[1]
        
        D_left = torch.pow(x_g_x - y_g_x,2)
        D_right = torch.pow(x_g_y - y_g_y,2)
        
        E = (D_left + D_right)
        
        return E

