import torch
import torch.nn.functional as F

from .FDN_arch import FDN
from .LPNet_arch import I_predict_net


class FNetColorExtractor:
    def __init__(self, fnet_ckpt_path, lpnet_ckpt_path, device):
        self.device = device
        self.net_ipred = I_predict_net().to(device)
        load_net_i = torch.load(lpnet_ckpt_path, map_location=device)
        self.net_ipred.load_state_dict(load_net_i["params"], strict=True)
        self.net_ipred.eval()

        self.net = FDN().to(device)
        load_net = torch.load(fnet_ckpt_path, map_location=device)
        self.net.load_state_dict(load_net["params"], strict=True)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, img_tensor):
        b, c, h, w = img_tensor.shape

        h_n = (32 - h % 32) % 32
        w_n = (32 - w % 32) % 32
        img_pad = F.pad(img_tensor, (0, w_n, 0, h_n), mode='reflect')

        ratio = self.net_ipred(img_pad)

        result, _, _, _ = self.net(img_pad, ratio_i=ratio, device=self.device)

        result = result[:, :, :h, :w]

        result = torch.clamp(result, 0.0, 1.0)

        return result