from transformers import logging
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers import logging
from diffusers.loaders import AttnProcsLayers
logging.set_verbosity_error()
from models.src.unet_hackd_lora import UNet2DConditionModel as UNet2DConditionModel_lora
from models.src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from models.src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
import random
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import AutoencoderKL,DDIMScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from diffusers.utils import logging

logging.set_verbosity(40)
from torchvision import transforms

from diffusers.image_processor import VaeImageProcessor


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)


class StableDiffusion(nn.Module):
    def __init__(self, opt, device, base_path, sd_version='2.0'):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(
                f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.opt = opt
        self.device = device
        self.sd_version = sd_version

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * self.opt.sd_max_step_start)
        self.weight_dtype = torch.float16
        print(f'[INFO] loading stable diffusion...')

        unet = UNet2DConditionModel.from_pretrained(
            # '/home/jian/IDM-train2/idm_plus_output_up/checkpoint-100/unet/',
            # "/home/jian/IDM-train2/idm_plus_output_up/checkpoint-54000",
            base_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        )

        tokenizer_one = AutoTokenizer.from_pretrained(
            'yisol/IDM-VTON',
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            'yisol/IDM-VTON',
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        noise_scheduler = DDIMScheduler.from_pretrained('yisol/IDM-VTON', subfolder="scheduler")
        # noise_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=self.weight_dtype)

        text_encoder_one = CLIPTextModel.from_pretrained(
            'yisol/IDM-VTON',
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            'yisol/IDM-VTON',
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'yisol/IDM-VTON',
            subfolder="image_encoder",
            torch_dtype=self.weight_dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            base_path,
            # "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        )
        # vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",
        #                                     subfolder="vae",
        #                                     torch_dtype=self.weight_dtype,
        #                                     )

        # "stabilityai/stable-diffusion-xl-base-1.0",
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            "yisol/IDM-VTON",
            subfolder="unet_encoder",
            torch_dtype=self.weight_dtype,
        )

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        unet.eval()
        UNet_Encoder.eval()

        self.unet = unet.to(self.device, self.weight_dtype)
        self.UNet_Encoder = UNet_Encoder.to(self.device, self.weight_dtype)
        self.vae = vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_one = text_encoder_one.to(self.device, self.weight_dtype)
        self.text_encoder_two = text_encoder_two.to(self.device, self.weight_dtype)
        self.image_encoder = image_encoder.to(self.device, self.weight_dtype)
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.noise_scheduler = noise_scheduler

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False, do_binarize=True,
                                                do_convert_grayscale=True
                                                )
        self.sds = opt.sds
        if not self.sds:
            _unet = UNet2DConditionModel_lora.from_pretrained(
                # '/home/jian/IDM-train2/idm_plus_output_up/checkpoint-100/unet/',
                # "/home/jian/IDM-train2/idm_plus_output_up/checkpoint-54000",
                base_path,
                subfolder="unet",
                torch_dtype=self.weight_dtype,
                low_cpu_mem_usage = False, device_map = None
            )
            _unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in _unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = _unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = _unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size,
                                                          cross_attention_dim=cross_attention_dim)
            _unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(_unet.attn_processors)

            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32

                def forward(self,latents_noisy,mask,masked_image_latents,pose_img_latents,
                            timesteps, text_embeddings, added_cond_kwargs, reference_features, c=None, shading="albedo",**kwargs):

                    latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents, pose_img_latents], dim=1)

                    pred = self.unet(
                                latent_model_input,
                                timesteps,
                                encoder_hidden_states=text_embeddings,
                                cross_attention_kwargs=None,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                                garment_features=reference_features,
                                c = c,
                                shading = shading,
                            )[0]
                    return pred

            self._unet = _unet
            self.lora_layers = lora_layers
            self.q_unet = LoraUnet().to(device)


            # self.parsing_model = Parsing(0)
        # self.openpose_model = OpenPose(0)
        # self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        self.alphas = self.noise_scheduler.alphas_cumprod.to(self.device)  # for conve

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        text = ''
        print(f'[INFO] preprocess text={text} !')
        (self.prompt_embeds,
         self.negative_prompt_embeds,
         self.pooled_prompt_embeds,
         self.negative_pooled_prompt_embeds,
         self.prompt_embeds_c) = self.get_text_prompt(text)
        self.normal = transforms.Normalize([0.5], [0.5])
        self.cfg=2.0
        print(f'[INFO] loaded IDM-TRY-ON diffusion!')

    def train_step(self, pred_rgb, ori_rgb, cloth_f, cloth_b, clip_f, clip_b, mask, pose_img,key_points, text="", ratio=0.02,
                   guidance_scale=2, lamb=1., pose=None, mask_p=None, global_step=0):

        # mask, pose_img = self.get_addition(transforms.ToPILImage()(pred_rgb[0]))
        # 0.数据预处理
        cloth = cloth_b if key_points[2, 0] > key_points[5, 0] else cloth_f
        clip = clip_b if key_points[2, 0] > key_points[5, 0] else clip_f
        clip = clip.to(self.device, self.weight_dtype)

        cloth = cloth.to(self.device, self.weight_dtype)
        clip = clip.to(self.device, self.weight_dtype)
        mask = mask.to(self.device, self.weight_dtype).unsqueeze(0)
        mask_p = mask_p.to(self.device, self.weight_dtype).unsqueeze(0)
        pose_img = pose_img.to(self.device, self.weight_dtype).unsqueeze(0)

        img_emb_list = []
        for i in range(clip.shape[0]):
            img_emb_list.append(clip[i])
        image_embeds = torch.cat(img_emb_list, dim=0).to(self.device)

        cloth = self.normal(cloth)
        init_image = self.normal(pred_rgb.unsqueeze(0))
        init_rgb = self.normal(ori_rgb.unsqueeze(0))

        if global_step > 1500: #800
            min_step_percent = 0.02
            max_step_percent = 0.35

        elif global_step > 1000: # 500
            min_step_percent = 0.02
            max_step_percent = 0.55

        elif global_step > 0:# 200
            min_step_percent = 0.02
            max_step_percent = 0.8
        # else:
        #     min_step_percent = 0.02
        #     max_step_percent = 0.98

        # min_step_percent= 0.02
        # max_step_percent= 0.98
        batch_size = init_image.shape[0]
        timesteps = torch.randint(
            int(self.noise_scheduler.config.num_train_timesteps * min_step_percent),  # 0.02
            int(self.noise_scheduler.config.num_train_timesteps * max_step_percent) + 1,
            (batch_size,), device=self.device
        )

        latents = self.encode_image(init_image.to(self.weight_dtype))

        with torch.no_grad():
            noise = torch.randn_like(latents)
            guidance_out = self.g(
                latents,
                noise,
                init_rgb.to(self.weight_dtype),
                mask,
                pose_img,
                cloth,
                image_embeds,
                timesteps,
                scale=guidance_scale,
            )
        latents_noisy  = guidance_out['latents_noisy']
        noise_pred= guidance_out['noise_pred']

        # mask = guidance_out["mask"][0]
        flag = False
        if flag:
            img = self.guidance_eval(init_image=init_image,
                noise=noise,
                init_rgb=init_rgb,
                pose_img=pose_img,
                cloth=cloth,
                image_embeds=image_embeds,
                timesteps=timesteps,
                scale=guidance_scale,mask1=mask, cfg=self.cfg,**guidance_out)
        mask_p = torch.nn.functional.interpolate(
            mask_p, size=(1024 //8, 768 // 8)
        )
        if not self.sds :
            if self.q_unet is not None:
                rand = random.random()
                if rand > 0.8:
                    shading = 'albedo'
                elif rand > 0.4 :
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

                if pose is not None:
                    pose = torch.from_numpy(pose).view(1, 16).to(self.device,self.weight_dtype)

                    noise_pred_q = self.q_unet(**guidance_out, timesteps=timesteps,
                                               c = pose, shading = shading)[0]
                else:
                    raise NotImplementedError()


                sqrt_alpha_prod = self.noise_scheduler.alphas_cumprod.to(self.device)[timesteps] ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                sqrt_one_minus_alpha_prod = (1 - self.noise_scheduler.alphas_cumprod.to(self.device)[timesteps]) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy


        # w(t), sigma_t^2
        w = (1 - self.alphas[timesteps])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        grad = w * (noise_pred - noise)
        grad = lamb * grad.clamp(-1, 1)

        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        return loss

    def get_add_time_ids(self, original_size, crops_coords_top_left=(0, 0)):
        add_time_ids = list(original_size + crops_coords_top_left + original_size)
        add_neg_time_ids = list(original_size + crops_coords_top_left + original_size)

        add_time_ids = torch.tensor([add_time_ids]).to(self.device, self.weight_dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids]).to(self.device, self.weight_dtype)
        return add_time_ids, add_neg_time_ids

    def get_text_prompt(self, text):
        prompt = "model is wearing " + text
        neg_text = "monochrome, lowres, bad anatomy, worst quality, low quality"
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            negative_prompt=neg_text,
        )
        prompt = "a photo of " + text
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        (
            prompt_embeds_c,
            _,
            _,
            _,
        ) = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, prompt_embeds_c

    def encode_prompt(self, prompt, negative_prompt):
        tokenizers = [self.tokenizer_one, self.tokenizer_two]
        text_encoders = (
            [self.text_encoder_one, self.text_encoder_two]
        )
        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt] * 2
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [negative_prompt] * 2
        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(self.device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            negative_prompt_embeds_list.append(negative_prompt_embeds)
        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_one.dtype, device=self.device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_two.dtype, device=self.device)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def encode_image(self, image):
        image = image.to(memory_format=torch.contiguous_format).float()
        image = image.to(self.vae.device, dtype=self.vae.dtype)
        return self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor


    def decode_latents(
            self,
            latents,
    ):
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def tokenize_caption(self, text):
        text_inputs = self.tokenizer_one(
            text, max_length=self.tokenizer_one.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        text_inputs_2 = self.tokenizer_two(
            text, max_length=self.tokenizer_two.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return text_inputs.input_ids, text_inputs_2.input_ids

    @torch.no_grad()
    def guidance_eval(
            self,
            init_image,
            noise,
            init_rgb,
            mask1,
            pose_img,
            cloth,
            image_embeds,
            timesteps,
            scale,
            t_orig,
            cloth_latent,
            masked_image_latents,
            pose_img_latents,
            mask,
            text_embeddings,
            text_embeds_cloth,
            noise_pred,
            added_cond_kwargs,
            cfg=1.0,
            **kwargs,
    ):
        # use only 50 timesteps, and find nearest of those to t
        # show.save(self.decode_latents(step_output["pred_original_sample"]*mask[0]+
        # (1-mask[0])*self.encode_image(init_rgb.unsqueeze(0))))
        self.noise_scheduler.set_timesteps(int(1000/timesteps*20))
        self.noise_scheduler.timesteps_gpu = self.noise_scheduler.timesteps.to(self.device)

        latents_final=[]
            # latents = latents_1step[b : b + 1]
            # latents = torch.randn_like(latents).to(self.weight_dtype)
        latents = self.encode_image(init_image.to(self.weight_dtype))
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            for t in self.noise_scheduler.timesteps[-20 - 1:-1]:
                noise_pred,_,noise_pred_uncond = self.get_noise_pred(
                    latents,
                    t,
                    cloth_latent,
                    masked_image_latents,
                    pose_img_latents,
                    mask,
                    text_embeddings,
                    self.prompt_embeds_c,
                    added_cond_kwargs,
                    cfg=2.0
                )

                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            # idxs = torch.min(large_enough_idxs, dim=1)[1]
            # t = self.guidance.noise_scheduler.timesteps_gpu[idxs]
            # for b in range(batch_size):
            #     step_output = self.guidance.noise_scheduler.step(
            #         noise_pred[b: b + 1], t[b], noisy_latents[b: b + 1], eta=0
            #     )
            # imgs_1orig = self.decode_latents(step_output["pred_original_sample"])
            imgs_final = self.decode_latents(latents)
        # show.show(imgs_final)
        return {
            "imgs_final": imgs_final,
        }

    def get_noise_pred(
            self,
            latents_noisy,
            t,
            cloth_latent,
            masked_image_latents,
            pose_img,
            mask,
            text_embeddings,
            text_embeds_cloth,
            added_cond_kwargs,
            cfg=1.0
    ):
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents, pose_img], dim=1)
        down, reference_features = self.UNet_Encoder(cloth_latent, t, text_embeds_cloth, return_dict=False)

        reference_features = list(reference_features)

        reference_features = [torch.cat([torch.zeros_like(d), d]) for d in reference_features]

        # print(f"{latent_model_input.shape}")
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            garment_features=reference_features,
        )[0]
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

        return noise_pred, reference_features,noise_pred_uncond
    def g(
            self,
            latents,
            noise,
            init_image,
            mask,
            pose_img,
            cloth,
            clip,
            timesteps,
            scale,
    ):
        latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        latents_noisy = latents
        cloth_latents = self.encode_image(cloth)  ##
        pose_latents = self.encode_image(pose_img)
        masked_image = init_image * (mask < 0.5)
        mask_image_latents = self.encode_image(masked_image)  ##

        mask = F.interpolate(mask.to(torch.float32), size=(int(1024 / 8), int(768 / 8)))
        mask = mask.to(self.device, dtype=self.weight_dtype)

        # Sample a random timestep for each image

        add_text_embeds = self.pooled_prompt_embeds
        # add cond
        add_time_ids = [
            torch.tensor([1024, 768]).to(self.device),
            torch.tensor([0, 0]).to(self.device),
            torch.tensor([1024, 768]).to(self.device),
        ]
        add_time_ids = torch.cat(add_time_ids).to(self.device, dtype=self.weight_dtype)

        img_emb_list = []
        for i in range(clip.shape[0]):
            img_emb_list.append(clip.unsqueeze(0)[i])
        image_embeds = torch.cat(img_emb_list, dim=0).to(self.device)

        with torch.no_grad():
            cloth_image_embeds = self.image_encoder(image_embeds.to(self.device, dtype=self.weight_dtype),
                                                    output_hidden_states=True).hidden_states[-2]
            cloth_image_embeds = cloth_image_embeds.repeat_interleave(1, dim=0)
            uncond_image_enc_hidden_states = \
                self.image_encoder(torch.zeros_like(image_embeds).to(self.device, dtype=self.weight_dtype),
                                   output_hidden_states=True).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(1, dim=0)

            # if cfg:
            cloth_image_embeds = torch.cat([uncond_image_enc_hidden_states, cloth_image_embeds])

        cloth_image_embeds = self.unet.encoder_hid_proj(
            cloth_image_embeds.to(self.device, dtype=self.weight_dtype))
        cloth_image_embeds = cloth_image_embeds.to(self.device, dtype=self.weight_dtype)
        prompt_embeds = self.prompt_embeds

        # if cfg:
        mask = torch.cat([mask] * 2)
        mask_image_latents = torch.cat([mask_image_latents] * 2)
        pose_latents = torch.cat([pose_latents] * 2)

        prompt_embeds = torch.cat([self.negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([self.negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        added_cond_kwargs["image_embeds"] = cloth_image_embeds

        noise_pred,reference_features,noise_pred_uncond = self.get_noise_pred(
            latents,
            timesteps,
            cloth_latents,
            mask_image_latents,
            pose_latents,
            mask,
            prompt_embeds,
            self.prompt_embeds_c,
            added_cond_kwargs,
            cfg=scale
        )

        return {
            "mask": mask,
            "masked_image_latents": mask_image_latents,
            "pose_img_latents": pose_latents,
            "t_orig": timesteps,
            "cloth_latent": cloth_latents,
            "text_embeddings": prompt_embeds,
            "text_embeds_cloth": self.prompt_embeds_c,
            "noise_pred": noise_pred,
            "added_cond_kwargs": added_cond_kwargs,
            'latents_noisy':latents_noisy,
            "encoder_hidden_states":prompt_embeds,
            "reference_features":reference_features,
            'noise_pred_uncond':noise_pred_uncond,
        }
