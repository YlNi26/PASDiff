import os
import torch
import torch.nn.functional as F
from safetensors import safe_open
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from bfr_model.lq_embed import vqvae_encoder, TwoLayerConv1x1
from utils.others import get_x0_from_noise
from guided_diffusion import dist_util


class BFREnhancer:
    def __init__(self, args):
        self.args = args
        self.device = dist_util.dev()

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        self.noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

        if args.merge_lora:
            self.unet = self.merge_unet(args)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

        self.img_encoder = vqvae_encoder(args).to(self.device, dtype=self.weight_dtype)

        if not args.cat_prompt_embedding:
            self.embedding_change = TwoLayerConv1x1(512, 1024)
            emb_path = os.path.join(args.ckpt_path, "embedding_change_weights.pth")
            if os.path.exists(emb_path):
                self.embedding_change.load_state_dict(torch.load(emb_path, map_location='cpu'))
            else:
                print(f"Warning: {emb_path} not found, initializing randomly.")
            self.embedding_change.to(self.device, dtype=self.weight_dtype)

        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder.to(self.device, dtype=self.weight_dtype)

        self.unet.eval()
        self.vae.eval()
        self.img_encoder.eval()

        self.timesteps = 399

    def merge_unet(self, args):
        print("Merging LoRA weights into UNet...")
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        lora_alpha = args.lora_alpha
        lora_rank = args.lora_rank
        alpha = float(lora_alpha / lora_rank)
        processed_keys = set()

        lora_path = os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")

        with safe_open(lora_path, framework="pt") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}

        state_dict_unet = unet.state_dict()

        for key in state_dict.keys():
            if "lora_A" in key:
                lora_a_key = key
                lora_b_key = key.replace("lora_A", "lora_B")
                unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.", "")

                if lora_b_key in state_dict and unet_key in state_dict_unet:
                    W_A = state_dict[lora_a_key]
                    W_B = state_dict[lora_b_key]
                    original_weight = state_dict_unet[unet_key]
                    processed_keys.update([lora_a_key, lora_b_key])

                    if len(original_weight.shape) == 4:
                        out_channels, in_channels, kH, kW = original_weight.shape
                        rank = W_A.shape[0]
                        W_A_flat = W_A.view(rank, -1)
                        W_B_flat = W_B.view(out_channels, rank)
                        delta_W_flat = torch.matmul(W_B_flat, W_A_flat)
                        delta_W = delta_W_flat.view(out_channels, in_channels, kH, kW)
                        merged_weight = original_weight + alpha * delta_W
                    else:
                        merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
                    state_dict_unet[unet_key] = merged_weight

            elif 'lora.up.weight' in key:
                lora_up_key = key
                lora_down_key = key.replace('lora.up.weight', 'lora.down.weight')
                original_weight_key = key.replace('.lora.up.weight', '.weight').replace("unet.", "")

                if lora_down_key in state_dict and original_weight_key in state_dict_unet:
                    W_up = state_dict[lora_up_key]
                    W_down = state_dict[lora_down_key]
                    W_orig = state_dict_unet[original_weight_key]
                    processed_keys.update([lora_up_key, lora_down_key])

                    delta_W = torch.matmul(W_up, W_down)
                    state_dict_unet[original_weight_key] = W_orig + alpha * delta_W

        unet.load_state_dict(state_dict_unet)
        print("Merge Done!")
        return unet

    @torch.no_grad()
    def enhance(self, img_tensor):
        lq = img_tensor * 2 - 1
        lq = lq.to(self.device, dtype=self.weight_dtype)

        lq_resized = F.interpolate(lq, (512, 512), mode='bilinear', align_corners=True)

        prompt_embeds = self.img_encoder(lq_resized).reshape(lq_resized.shape[0], 77, -1)
        if not self.args.cat_prompt_embedding:
            prompt_embeds = self.embedding_change(prompt_embeds)

        lq_latent = self.vae.encode(lq_resized).latent_dist.sample() * self.vae.config.scaling_factor

        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample

        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            self.timesteps
        ).float()

        output_image = self.vae.decode(x_0.to(self.weight_dtype) / self.vae.config.scaling_factor).sample
        output_image = output_image.clamp(-1, 1)
        output_image = output_image * 0.5 + 0.5

        return output_image