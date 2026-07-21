import argparse
import os
import sys
import collections
import cv2
import os.path as osp
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torch
import time
sys.path.append(os.getcwd())

from guided_diffusion import dist_util, logger
from utils.Attribute import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from bfr_model.bfr_enhancer import  BFREnhancer


def main(inference_step=None):
    L_exp = L_exp2(1)

    args = create_argparser().parse_args()
    dist_util.setup_dist()

    print('===> Building BFR model')
    enhancer = BFREnhancer(args)
    print('BFR enhancer is ready.')

    out_dir = f'{args.out_dir}/s{args.guidance_scale}_dw{args.deblur_weight}_cw{args.color_map_weight}_ew{args.exposure_weight}_be{args.base_exposure}_aa{args.adjustment_amplitude}_seed{args.seed}'
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(dist_util.dev())
    model.eval()

    print(f'===> Building {args.retinex_type.upper()} retinex model')
    if args.retinex_type == 'rnet':
        from retinex_model.Rnet import net as Rnet
        retinex_model = Rnet().to(dist_util.dev())
        if args.retinex_model.split('.')[-1] == 'ckpt':
            print("Loading Rnet checkpoint")
            ckpt = torch.load(args.retinex_model, map_location=lambda storage, loc: storage)
            new_state_dict = collections.OrderedDict()
            for k in ckpt['state_dict']:
                if k[:6] != 'model.':
                    continue
                name = k[6:]
                new_state_dict[name] = ckpt['state_dict'][k]
            retinex_model.load_state_dict(new_state_dict, strict=True)
        else:
            retinex_model.load_state_dict(torch.load(args.retinex_model, map_location=lambda storage, loc: storage))
    elif args.retinex_type == 'real':
        from retinex_model.FNet.fnet_wrapper import FNetColorExtractor
        retinex_model = FNetColorExtractor(fnet_ckpt_path=args.fnet_model_path, lpnet_ckpt_path=args.lpnet_model_path, device=dist_util.dev())
    else:
        raise ValueError(f"Unsupported retinex_type: {args.retinex_type}")

    print(f'Pre-trained {args.retinex_type.upper()} retinex model is loaded.')
    if hasattr(retinex_model, 'eval'):
        retinex_model.eval()

    print("=================== Summary (Sampling) ===================")
    print(f'Task: {args.task}; Guidance scale: {args.guidance_scale}')
    print("==========================================================")

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    def attribute_guidance(x, t, y=None, pred_xstart=None, target=None, ref=None, mask=None,
                           task="LIE", scale=0, N=None, exposure_map=None, reflectence_map=None):
        assert y is not None
        with th.enable_grad():
            predicted_start = pred_xstart.detach().requires_grad_(True)
            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            predicted_start_norm = ((predicted_start + 1) * 0.5)
            target_norm = ((y + 1) * 0.5)

            illumination_loss = L_exp(predicted_start_norm, exposure_map) * args.exposure_weight
            reflectance_loss = F.mse_loss(reflectence_map, predicted_start_norm, reduction='sum') * args.color_map_weight
            face_deblur_loss = th.tensor(0.0, device=predicted_start.device)

            if enhancer is not None:
                with th.no_grad():
                    b, c, h, w = predicted_start_norm.shape

                    sr_target = enhancer.enhance(predicted_start_norm)
                    sr_target = sr_target.to(predicted_start.dtype)

                    if sr_target.shape[2] != h or sr_target.shape[3] != w:
                        sr_target = F.interpolate(sr_target, size=(h, w), mode='area')
                pred_mean = predicted_start_norm.mean(dim=(2, 3), keepdim=True)
                pred_std = predicted_start_norm.std(dim=(2, 3), keepdim=True)
                target_mean = sr_target.mean(dim=(2, 3), keepdim=True)
                target_std = sr_target.std(dim=(2, 3), keepdim=True)
                sr_target_aligned = (sr_target - target_mean) / (target_std + 1e-5) * pred_std + pred_mean

                face_deblur_loss = F.mse_loss(predicted_start_norm, sr_target_aligned.detach()) * args.deblur_weight

            total_loss = illumination_loss + reflectance_loss + face_deblur_loss

            print(f'loss (exp): {illumination_loss:.4f};', end=' ')
            print(f'loss (col): {reflectance_loss:.4f};', end=' ')
            print(f'loss (deblur): {face_deblur_loss:.4f};', end=' ')
            print(f'loss (tot): {total_loss:.4f};')

            gradient = th.autograd.grad(total_loss, predicted_start)[0]

        return gradient, None

    def model_fn(x, t, y=None, target=None, ref=None, mask=None,
                 task=None, scale=0, N=1, exposure_map=None, reflectence_map=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    all_images = []
    lr_folder = args.in_dir
    lr_images = sorted(os.listdir(lr_folder))

    logger.log("Sampling...")
    total_images = 0

    for img_name in lr_images:
        path_lq = osp.join(lr_folder, img_name)
        raw = cv2.imread(path_lq).astype(np.float32)[:, :, [2, 1, 0]]
        y00 = th.as_tensor(raw / 255).permute(2, 0, 1).unsqueeze(0).to(dist_util.dev())
        y0 = th.tensor(raw / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(dist_util.dev())

        print(img_name)
        _, _, H, W = y0.shape
        
        if args.retinex_type == 'rnet':
            current_reflectence_map = calculate_color_map(y00, retinex_model)
        elif args.retinex_type == 'real':
            current_reflectence_map = retinex_model(y00)

        model_kwargs = {
            "task": args.task,
            "target": None,
            "scale": args.guidance_scale,
            "N": args.N,
            "exposure_map": check_image_size(calculate_spatially_varying_exposure_HSI(path_lq, args.base_exposure, args.adjustment_amplitude)),
            "y": check_image_size(y0),
            "reflectence_map": check_image_size(current_reflectence_map)
        }
        b, c, h, w = model_kwargs["y"].shape

        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, h, w),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=attribute_guidance,
            device=dist_util.dev(),
            seed=seed,
            inference_step=inference_step
        )

        total_images += 1
        sample = ((sample[:, :, :H, :W] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(f'{out_dir}/{img_name}', all_images[-1][0][..., [2, 1, 0]])
        torch.cuda.empty_cache()



    dist.barrier()
    logger.log("Sampling complete!")


def create_argparser():
    defaults = dict(
        seed=12345678,
        task='LIE',
        in_dir='./inputs',
        out_dir='./outputs',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./checkpoints/256x256_diffusion_uncond.pt", 
        retinex_type='rnet', 
        retinex_model="./checkpoints/RNet_1688_step.ckpt",
        fnet_model_path="./checkpoints/FDN_lolblur.pth",
        lpnet_model_path="./checkpoints/LPNet_lolblur.pth",   
        pretrained_model_name_or_path='./checkpoints/stable-diffusion-2-1-base',
        ckpt_path='./checkpoints',
        img_encoder_weight='./checkpoints/associate_2.ckpt',
        guidance_scale=2.3,
        color_map_weight=0.03,
        exposure_weight=1200,
        deblur_weight=10000,
        base_exposure=0.46,
        adjustment_amplitude=0.25,
        N=5,
        mixed_precision='fp16',
        merge_lora=True,
        lora_rank=16,
        lora_alpha=16,
        cat_prompt_embedding=False,
        use_att_pool=False,
        use_pos_embedding=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main(inference_step=10)