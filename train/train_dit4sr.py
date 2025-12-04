
from diffusers.optimization import get_scheduler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())
from accelerate.logging import get_logger

import cv2 
import wandb
# import pyiqa
import math
import argparse
import numpy as np 
from PIL import Image 
from tqdm.auto import tqdm
from einops import rearrange 
from omegaconf import OmegaConf

from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

import initialize 
import train_utils
from dataloaders.utils import encode, decode
from train_utils import encode_prompt, get_sigmas
from dataloaders.utils import realesrgan_degradation
from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline

import torch
import torchvision.transforms as T 
import torch.nn.functional as F 
from torchvision.utils import save_image 

from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix


logger = get_logger(__name__)


def main(cfg):

    
    # set experiment name
    # exp_name = f'{cfg.train.mixed_precision}__{cfg.train.stage}__{"-".join(cfg.train.model)}__{"-".join(f"{lr:.0e}" for lr in cfg.train.lr)}__bs-{str(cfg.train.batch_size)}__gradaccum-{cfg.train.gradient_accumulation_steps}__{"-".join(cfg.train.finetune)}__ocrloss{cfg.train.ocr_loss_weight}__{cfg.model.dit.text_condition.caption_style}__{cfg.log.tracker.msg}'
    exp_name = f'{cfg.train.mixed_precision}__{cfg.train.stage}'
    if 'transformer' in cfg.train.model:
        exp_name = f'{exp_name}__{cfg.train.transformer.architecture}-{cfg.train.transformer.lr:.0e}'
        if cfg.train.transformer.ocr_branch_init is not None:
            exp_name = f'{exp_name}__ocrbranchinit-{cfg.train.transformer.ocr_branch_init}'
    if 'ts_module' in cfg.train.model:
        exp_name = f'{exp_name}__{cfg.train.ts_module.architecture}-{cfg.train.ts_module.lr:.0e}__ocrloss{cfg.train.ocr_loss_weight}__extract-{cfg.train.transformer.feat_extract}-num-{str(len(cfg.train.transformer.feat_extract_layer))}'
    exp_name = f'{exp_name}__bs-{str(cfg.train.batch_size)}__gradaccum-{cfg.train.gradient_accumulation_steps}__{cfg.log.tracker.msg}'
    cfg.exp_name = exp_name
    print(f'-' * 50)
    print(f'Experiment name: {exp_name}')
    print(f'-' * 50)
    
    
    # set accelerator and basic settings (seed, logging, dir_path)
    accelerator = initialize.load_experiment_setting(cfg, logger, exp_name)
    
    
    # set tracker
    initialize.load_trackers(cfg, accelerator, exp_name)


    # load data
    train_dataloader, val_datasets = initialize.load_data(cfg)

    
    # load models 
    models = initialize.load_model(cfg, accelerator)


    # load model parameters (total_params, trainable_params, frozen_params)
    model_params = initialize.load_model_params(cfg, accelerator, models)


    # load optimizer 
    optimizer = initialize.load_optim(cfg, accelerator, models)


    # place models on cuda and proper weight dtype(float32, float16)
    weight_dtype = initialize.set_model_device(cfg, accelerator, models)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if cfg.train.max_train_steps is None:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.train.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.train.lr_num_cycles,
        power=cfg.train.lr_power,
    )


    # Prepare everything with our `accelerator`.
    # transformer, ts_module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], models['testr'], optimizer, train_dataloader, lr_scheduler)
    # transformer, models['testr'], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], models['testr'], optimizer, train_dataloader, lr_scheduler)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], optimizer, train_dataloader, lr_scheduler)
    vae_img_processor = VaeImageProcessor(vae_scale_factor=8)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.train.num_train_epochs = math.ceil(cfg.train.max_train_steps / num_update_steps_per_epoch)


    # # SR metrics
    # metric_psnr = pyiqa.create_metric('psnr', device=accelerator.device)
    # metric_ssim = pyiqa.create_metric('ssimc', device=accelerator.device)
    # metric_lpips = pyiqa.create_metric('lpips', device=accelerator.device)
    # metric_dists = pyiqa.create_metric('dists', device=accelerator.device)
    # # metric_fid = pyiqa.create_metric('fid', device=device)
    # metric_niqe = pyiqa.create_metric('niqe', device=accelerator.device)
    # metric_musiq = pyiqa.create_metric('musiq', device=accelerator.device)
    # metric_maniqa = pyiqa.create_metric('maniqa', device=accelerator.device)
    # metric_clipiqa = pyiqa.create_metric('clipiqa', device=accelerator.device)


    tot_train_epochs = cfg.train.num_train_epochs
    tot_train_steps = cfg.train.max_train_steps


    # Train!
    total_batch_size = cfg.train.batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("=== Model Parameters ===")
    logger.info(f"  Total Params    : {model_params['tot_param']:,} ({model_params['tot_param']/1e6:.2f}M)")
    logger.info(f"  Trainable Params: {model_params['train_param']:,} ({model_params['train_param']/1e6:.2f}M)")
    logger.info(f"  Frozen Params   : {model_params['frozen_param']:,} ({model_params['frozen_param']/1e6:.2f}M)")

    logger.info("=== Training Setup ===")
    logger.info(f"  Num training samples = {len(train_dataloader.dataset)}")
    # logger.info(f"  Num validation samples = {len(val_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {tot_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {tot_train_steps}")

    # logger.info("=== Parameter Names ===")
    # logger.info(f"  Frozen Params ({len(model_params['frozen_param_names'])}):")
    # for name in model_params['frozen_param_names']:
    #     logger.info(f" FROZEN - {name}")
    # logger.info(f"  Trainable Params ({len(model_params['train_param_names'])}):")
    # for name in model_params['train_param_names']:
    #     logger.info(f" TRAINING - {name}")
    
    
    # save trainable params as txt 
    if accelerator.is_main_process:
        txt_file = f'{cfg.save.output_dir}/{exp_name}/train_params.txt'
        with open(txt_file, 'w') as f:
            # log trainable params 
            for name in model_params['train_param_names']:
                f.write(f'TRAINABLE - {name}\n')
            # log frozen params 
            for name in model_params['frozen_param_names']:
                f.write(f'FROZEN - {name}\n')
        # img_file = train_utils.text_file_to_image(txt_file)


    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    ocr_losses={}  

    progress_bar = tqdm(range(0, tot_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)
    free_memory()
    for epoch in range(first_epoch, tot_train_epochs):
        for step, batch in enumerate(train_dataloader):

            if cfg.data.train.name == 'satext':
                batch = realesrgan_degradation(batch)

                gt = batch['gt']
                lq = batch['lq']
                text = batch['text']
                text_encs = batch['text_enc']
                hq_prompt = batch['hq_prompt']
                # lq_prompt = batch['lq_prompt']
                boxes = batch['bbox']    # len(bbox) = b
                polys = batch['poly']    # len(poly) = b
                img_id = batch['img_id']
            

            with accelerator.accumulate([transformer]):

                if cfg.data.train.name == 'satext':
                    with torch.no_grad():
                        # hq vae encoding
                        gt = gt.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                        hq_latents = models['vae'].encode(gt).latent_dist.sample()  # b 16 64 64
                        model_input = (hq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor    # b 16 64 64 
                        model_input = model_input.to(dtype=weight_dtype)
                        # lq vae encoding
                        lq = lq.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                        lq_latents = models['vae'].encode(lq).latent_dist.sample()  # b 16 64 64 
                        controlnet_image = (lq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor   # b 16 64 64 
                        controlnet_image = controlnet_image.to(dtype=weight_dtype)
                        # # load caption
                        # if cfg.model.dit.load_precomputed_caption:
                        #     hq_prompt = hq_prompt 
                        # else:
                        #     # vlm captioner
                        #     lq_tmp = F.interpolate(lq, size=(336, 336), mode="bilinear", align_corners=False)
                        #     hq_prompt = models['vlm_agent'].gen_image_caption(lq_tmp)
                        #     hq_prompt = [train_utils.remove_focus_sentences(p) for p in hq_prompt]

                        # set prompt style
                        if cfg.model.dit.use_gtprompt:
                            if cfg.model.dit.text_condition.caption_style == 'descriptive':
                                texts = [[f'"{t}"' for t in txt] for txt in text]
                                hq_prompt = [f'The image features the texts {", ".join(txt)} that appear clearly on signs, boards, buildings, or other objects.' for txt in texts]
                            elif cfg.model.dit.text_condition.caption_style == 'tag':
                                texts = [[f'"{t}"' for t in txt] for txt in text]
                                hq_prompt=[', '.join(words) for words in texts]  

                        # encode prompt 
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(models['text_encoders'], models['tokenizers'], hq_prompt, 77)
                        prompt_embeds = prompt_embeds.to(model_input.dtype)                 # b 154 4096
                        pooled_prompt_embeds = pooled_prompt_embeds.to(model_input.dtype)   # b 2048
                else:
                    model_input = batch["pixel_values"].to(dtype=weight_dtype)
                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)   # b 16 64 64 
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.model.noise_scheduler.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=cfg.model.noise_scheduler.logit_mean,
                    logit_std=cfg.model.noise_scheduler.logit_std,
                    mode_scale=cfg.model.noise_scheduler.mode_scale,
                )

                indices = (u * models['noise_scheduler_copy'].config.num_train_timesteps).long()
                timesteps = models['noise_scheduler_copy'].timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching. b
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, accelerator, models['noise_scheduler_copy'], n_dim=model_input.ndim, dtype=model_input.dtype)    # b 1 1 1
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise   # b 16 64 64 
                # with torch.cuda.amp.autocast(enabled=False):
                # Predict the noise residual
                trans_out = transformer(                       
                    hidden_states=noisy_model_input,            # b 16 64 64 
                    controlnet_image=controlnet_image,          # b 16 16 16  
                    timestep=timesteps,                         # b
                    encoder_hidden_states=prompt_embeds,        # b 154 4096
                    pooled_projections=pooled_prompt_embeds,    # b 2048
                    return_dict=False,
                    cfg=cfg
                )
                model_pred = trans_out[0]   # b 16 64 64

                if len(trans_out) > 1:
                    etc_out = trans_out[1]
                    # unpatchify
                    patch_size = models['transformer'].config.patch_size  # 2
                    hidden_dim = models['transformer'].config.num_attention_heads * models['transformer'].config.attention_head_dim     # 1536
                    height = 64 // patch_size       # 32
                    width = 64 // patch_size        # 32
                    

                    # -- only hq feat -- 
                    # 1 2048 1536 -> only bring the hq tokens -> 1 1024 1536
                    # extracted_feats = [ rearrange(feat['extract_feat'], 'b (H W) (pH pW d) -> b d (H pH) (W pW)', H=height, W=width, pH=patch_size, pW=patch_size) for feat in etc_out ]    # b 384 64 64 
                    
                    # -- hq + lq feat --
                    # 1 2048 1536 -> bring both hq and lq tokens -> 1 2 1024 1536
                    if cfg.train.transformer.feat_extract == 'hqlq_feat':
                        num_concat_feat = 2
                    else:
                        num_concat_feat = 1
                    extracted_feats = [ rearrange(feat['extract_feat'], 'b (N H W) (pH pW d) -> b (N d) (H pH) (W pW)', N=num_concat_feat, H=height, W=width, pH=patch_size, pW=patch_size) for feat in etc_out ]    # b 384 64 64 
                    
                    # if cfg.train.repa.use_repa_tsm:
                    #     # REPA - extract features from specific layers
                    #     extracted_feats = [feat for idx_feat, feat in enumerate(extracted_feats) if idx_feat in cfg.train.repa.tsm_applied_layer]
                    

                      

                '''
                (Pdb) extracted_feats[0].shape  torch.Size([2, 1280, 16, 16])
                (Pdb) extracted_feats[1].shape  torch.Size([2, 1280, 32, 32])
                (Pdb) extracted_feats[2].shape  torch.Size([2, 640, 64, 64])
                (Pdb) extracted_feats[3].shape  torch.Size([2, 320, 64, 64])

                (Pdb) trans_out[0]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                (Pdb) trans_out[1]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                (Pdb) trans_out[2]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                (Pdb) trans_out[3]['extract_feat'].shape    torch.Size([2, 1024, 1536])
                '''


                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if cfg.model.noise_scheduler.precondition_outputs:   # t
                    model_pred = model_pred * (-sigmas) + noisy_model_input # b 16 64 64

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=cfg.model.noise_scheduler.weighting_scheme, sigmas=sigmas)   # b 1 1 1

                # flow matching loss
                if cfg.model.noise_scheduler.precondition_outputs:   # t
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                diff_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                diff_loss = diff_loss.mean()

                # ts module loss 
                if 'ts_module' in cfg.train.model:
                    # process annotations for OCR training loss
                    train_targets=[]
                    for i in range(bsz):
                        num_box=len(boxes[i])
                        tmp_dict={}
                        tmp_dict['labels'] = torch.tensor([0]*num_box).to(accelerator.device)  # 0 for text
                        tmp_dict['boxes'] = torch.tensor(boxes[i]).to(accelerator.device)   # xyxy format, absolute coord, [num_box, 4]
                        tmp_dict['texts'] = text_encs[i]
                        tmp_dict['ctrl_points'] = polys[i]
                        train_targets.append(tmp_dict)

                    with torch.cuda.amp.autocast(enabled=False):
                        # OCR model forward pass
                        ocr_loss_dict, ocr_result = models['testr'](extracted_feats, train_targets, MODE='TRAIN')
                    
                    # OCR total_loss
                    ocr_tot_loss = sum(v for v in ocr_loss_dict.values())
                    # OCR losses
                    for ocr_key, ocr_val in ocr_loss_dict.items():
                        if ocr_key in ocr_losses.keys():
                            ocr_losses[ocr_key].append(ocr_val.item())
                        else:
                            ocr_losses[ocr_key]=[ocr_val.item()]
                #     total_loss = diff_loss + cfg.train.ocr_loss_weight * ocr_tot_loss      
                # else:
                #     total_loss = diff_loss
                #     ocr_tot_loss=torch.tensor(0).cuda()
                
                
                
                # -------------------------------------------
                #   Loss calculation for different stages
                # -------------------------------------------
                if cfg.train.stage == 'stage1':
                    total_loss = diff_loss
                    ocr_tot_loss=torch.tensor(0).cuda()
                
                elif cfg.train.stage == 'stage2':
                    total_loss = cfg.train.ocr_loss_weight * ocr_tot_loss
                
                elif cfg.train.stage == 'stage3':
                    total_loss = diff_loss + cfg.train.ocr_loss_weight * ocr_tot_loss      
                    
                    


                # Inside your training loop, after backprop and gradient clipping
                if global_step > 0:

                    # -----------------------------
                    # Backprop
                    # -----------------------------
                    accelerator.backward(total_loss)

                    # -----------------------------
                    # Clip gradients
                    # -----------------------------
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(list(transformer.parameters()), cfg.train.max_grad_norm)

                        if 'testr' in models and getattr(models['testr'], 'training', False):
                            torch.nn.utils.clip_grad_norm_(models['testr'].parameters(),
                                                        max_norm=cfg.train.max_grad_norm)

                    # -----------------------------
                    # Helper: get max grad
                    # -----------------------------
                    def get_max_grad(model):
                        max_grad = 0.0
                        for param in model.parameters():
                            if param.grad is not None:
                                max_grad = max(max_grad, param.grad.abs().max().item())
                        return max_grad

                    transformer_max_grad = get_max_grad(transformer)
                    testr_max_grad = get_max_grad(models['testr']) if 'testr' in models else 0.0

                    # -----------------------------
                    # Helper: get TOP-K grads as text
                    # -----------------------------
                    def get_topk_gradients(model, top_k=20):
                        grad_dict = {}
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_dict[name] = param.grad.abs().max().item()
                        sorted_grads = sorted(grad_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

                        txt = ""
                        for name, g in sorted_grads:
                            txt += f"{name:<60} {g:>10.6f}\n"
                        return txt

                    # -----------------------------
                    # WandB Logging
                    # -----------------------------
                    if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
                        wandb.log({
                            "gradients/transformer_max": transformer_max_grad,
                            "gradients/testr_max": testr_max_grad,
                            "loss/diff_loss": diff_loss.item(),
                            "loss/ocr_tot_loss": ocr_tot_loss.item(),
                            "loss/total_loss": total_loss.item(),
                        }, step=global_step)

                    # -----------------------------
                    # TXT Logging (main process)
                    # -----------------------------
                    if cfg.val.val_every_step == 0:
                        if accelerator.is_main_process:

                            save_norm_monitor_path = f"{cfg.save.output_dir}/{exp_name}/monitor_grad_norm"
                            os.makedirs(save_norm_monitor_path, exist_ok=True)

                            # Get top-K gradients as text
                            transformer_topk_txt = get_topk_gradients(transformer, top_k=20)

                            if 'testr' in models and getattr(models['testr'], 'training', False):
                                testr_topk_txt = get_topk_gradients(models['testr'], top_k=20)
                            else:
                                testr_topk_txt = "(testr missing or not training)\n"

                            # Build log string
                            log_str = (
                                f"[Step {global_step}]\n"
                                f"transformer_max_grad = {transformer_max_grad:.6f}\n"
                                f"testr_max_grad       = {testr_max_grad:.6f}\n"
                                f"diff_loss            = {diff_loss.item():.6f}\n"
                                f"ocr_tot_loss         = {ocr_tot_loss.item():.6f}\n"
                                f"total_loss           = {total_loss.item():.6f}\n"
                                f"{'-'*60}\n"
                                f"Top 20 Transformer Gradients:\n"
                                f"{transformer_topk_txt}"
                                f"{'-'*60}\n"
                                f"Top 20 Testr Gradients:\n"
                                f"{testr_topk_txt}"
                                f"{'='*80}\n\n"
                            )

                            with open(f"{save_norm_monitor_path}/grad_norm_step{global_step:09d}.txt", "a") as f:
                                f.write(log_str)

                    # -----------------------------
                    # Optimizer step
                    # -----------------------------
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=cfg.train.set_grads_to_none)



            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1



                if accelerator.is_main_process:
                    if global_step % cfg.save.checkpointing_steps == 0:
                        
                        # set save directory
                        save_path = os.path.join(cfg.save.output_dir, exp_name, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        # save transformer
                        if 'transformer' in cfg.train.model:
                            accelerator.save_state(save_path)
                        
                        # save text spotting module 
                        if 'ts_module' in cfg.train.model:
                            # save ts_module
                            ts_ckpt = {}
                            ts_ckpt['ts_module'] = models['testr'].state_dict()
                            ckpt_path = f"{save_path}/ts_module{global_step:07d}.pt"
                            torch.save(ts_ckpt, ckpt_path)
                        logger.info(f"Saved state to {save_path}")



                    if len(cfg.data.val.eval_list) != 0 and (global_step==1 or global_step % cfg.val.val_every_step == 0):

                        if 'ts_module' in cfg.train.model:
                            ts_module = models['testr'] 
                        else:
                            ts_module = None 
                        # load validation pipeline
                        val_pipeline = StableDiffusion3ControlNetPipeline(
                            vae=models['vae'], text_encoder=models['text_encoders'][0], text_encoder_2=models['text_encoders'][1], text_encoder_3=models['text_encoders'][2], 
                            tokenizer=models['tokenizers'][0], tokenizer_2=models['tokenizers'][1], tokenizer_3=models['tokenizers'][2], 
                            transformer=models['transformer'], scheduler=models['noise_scheduler'], ts_module=ts_module,
                        )


                        # val loop
                        for val_data_name, val_data in val_datasets.items():

                            for sample_idx, val_sample in enumerate(val_data):
                                
                                print('-------------------------------------------------')
                                print(f'{cfg.data.val.eval_region} - {val_data_name} - {sample_idx+1}/{len(val_data)} - using {cfg.data.val.text_cond_prompt}prompt') 
                                
                                generator = None
                                if accelerator.is_main_process and cfg.init.seed is not None:
                                    generator = torch.Generator(device=accelerator.device)
                                    generator.manual_seed(cfg.init.seed)


                                # load val anns
                                val_lq_path = val_sample['lq_path']
                                val_hq_path = val_sample['hq_path']
                                val_gt_text = val_sample['text']
                                val_bbox = val_sample['bbox']       # xyxy
                                val_polys = val_sample['poly']
                                val_img_id = val_sample['img_id']
                                val_vlm_cap = val_sample['vlm_cap']
                                
                                
                                # process lq image 
                                val_lq_pil = Image.open(val_lq_path).convert("RGB") # 128 128 
                                ori_width, ori_height = val_lq_pil.size
                                rscale = 4  # upscale x4
                                # for shortest side smaller than 128, resize
                                if ori_width < 512//rscale or ori_height < 512//rscale:
                                    scale = (512//rscale)/min(ori_width, ori_height)
                                    tmp_image = val_lq_pil.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)
                                    val_lq_pil = tmp_image
                                val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]*rscale, val_lq_pil.size[1]*rscale), Image.BICUBIC)
                                val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]//8*8, val_lq_pil.size[1]//8*8), Image.BICUBIC)
                                

                                
                                # -------------------------------------------------
                                #       input prompt to diffusion model
                                # -------------------------------------------------
                                
                                # gt prompt 
                                if cfg.data.val.text_cond_prompt == 'gt':
                                    texts = [f'"{t}"' for t in val_gt_text]
                                    if cfg.model.dit.text_condition.caption_style == 'descriptive':
                                        val_init_prompt = [f'The image features the texts {", ".join(texts)} that appear clearly on signs, boards, buildings, or other objects.']
                                    elif cfg.model.dit.text_condition.caption_style == 'tag':
                                        val_init_prompt = [f"{', '.join(texts)}"]

                                # text spotting module prompt
                                elif cfg.data.val.text_cond_prompt == 'pred_tsm':
                                    val_init_prompt = ['']
                                    
                                # vlm prompt 
                                elif cfg.data.val.text_cond_prompt == 'pred_vlm':
                                    val_init_prompt = [val_vlm_cap]
                                    
                                # null prompt 
                                elif cfg.data.val.text_cond_prompt == 'null':
                                    val_init_prompt = ['']
                                
                                else:
                                    val_init_prompt = ['']
                                    
                                
                                # added prompt             
                                if cfg.data.val.added_prompt is not None:
                                    val_init_prompt = [f'{val_init_prompt[0]} {cfg.data.val.added_prompt}']

                                # cfg negative prompt 
                                if cfg.data.val.negative_prompt is not None:
                                    neg_prompt = cfg.data.val.negative_prompt
                                else:
                                    neg_prompt = None 
                                    
                                    
                            
                                # -------------------------------------------------
                                #       validation pipeline forward pass 
                                # -------------------------------------------------
                                with torch.no_grad():
                                    val_out = val_pipeline(
                                        prompt=val_init_prompt[0], control_image=val_lq_pil, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=512, width=512,
                                        guidance_scale=cfg.data.val.guidance_scale, negative_prompt=neg_prompt,
                                        start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                                        output_type = 'pil', return_dict=False, lq_id=val_img_id, val_data_name=val_data_name, global_step=global_step, cfg=cfg, mode='train'
                                    )
                                

                                
                                # retrieve restored image 
                                val_res_pil = val_out[0][0]   # 1 3 512 512 [0,1]
                                
                                
                                # post processing
                                if cfg.data.val.align_method is not None:                
                                    if cfg.data.val.align_method == 'wavelet':
                                        val_res_pil = wavelet_color_fix(val_res_pil, val_lq_pil)
                                    elif cfg.data.val.align_method == 'adain':
                                        val_res_pil = adain_color_fix(val_res_pil, val_lq_pil)
                                

                                # # Save only the restored image
                                # val_res_save_path = f'{cfg.save.output_dir}/{exp_name}/{val_data_name}/final_restored_img'
                                # os.makedirs(val_res_save_path, exist_ok=True)
                                # val_res_pil.save(f'{val_res_save_path}/{val_img_id}.png')
                                

                                
                                # ---------------------------------------
                                #       image metric calculation
                                # ---------------------------------------
                                
                                # restored pil img -> tensor 
                                val_res_pt = T.ToTensor()(val_res_pil) 
                                val_res_pt = val_res_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
                                
                                # gt pil img -> tensor 
                                val_hq_pil = Image.open(val_hq_path).convert("RGB") 
                                val_hq_pt = T.ToTensor()(val_hq_pil)
                                val_hq_pt = val_hq_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
                                
                                # lq pil img -> tensor 
                                val_lq_pt = T.ToTensor()(val_lq_pil)
                                val_lq_pt = val_lq_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
                                
                                                             


                                # --------------------------------------------------
                                #       restoration and OCR visualization 
                                # --------------------------------------------------

                                # save path
                                val_res_ocr_save_path = f'{cfg.save.output_dir}/{exp_name}/{val_data_name}/final_result'
                                os.makedirs(val_res_ocr_save_path, exist_ok=True)
                                
                                
                                # hq img: pt -> np
                                vis_hq = val_hq_pt.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy() * 255.0
                                vis_hq = vis_hq.astype(np.uint8)
                                vis_hq1 = vis_hq.copy()
                                vis_hq2 = vis_hq.copy()
                                
                                # restored img: pt -> np
                                vis_res = val_res_pt.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy() * 255.0
                                vis_res = vis_res.astype(np.uint8)
                                
                                # lq img: pt -> np
                                vis_lq = val_lq_pt.squeeze(dim=0).permute(1,2,0).detach().cpu().numpy() * 255.0
                                vis_lq = vis_lq.astype(np.uint8)
                                vis_lq = vis_lq.copy()
                                
                                            
                                            
                                # vis ocr result
                                if ('ts_module' in cfg.train.model) and (cfg.data.val.ocr.vis_ocr):
                                    
                                    # prepare ocr visualization
                                    val_ocr_save_path = f'{cfg.save.output_dir}/{exp_name}/{val_data_name}/final_ocr_result'
                                    os.makedirs(val_ocr_save_path, exist_ok=True)
                                    
                                    
                                    print('== logging val ocr results ===')
                                    print(f'- Evaluating {val_data_name} - {val_img_id}')
                                    
                                    
                                    val_ocr_result = val_out[1]
                                    
                                    
                                    # ------------------ overlay gt ocr ------------------
                                    vis_hq1 = vis_hq1.copy()
                                    gt_polys = val_polys           # b 16 2
                                    gt_texts = val_gt_text
                                    for vis_img_idx in range(len(gt_polys)):
                                        gt_poly = gt_polys[vis_img_idx]   # 16 2
                                        gt_poly = gt_poly.astype(np.int32)
                                        gt_txt = gt_texts[vis_img_idx]
                                        cv2.polylines(vis_hq1, [gt_poly], isClosed=True, color=(0,255,0), thickness=2)
                                        cv2.putText(vis_hq1, gt_txt, (gt_poly[0][0], gt_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                    # ------------------ overlay gt ocr ------------------
                                    
                                    
                                    vis_result = cv2.hconcat([vis_hq1, vis_lq])
                                    # visualize ocr results per denoising timestep
                                    for ocr_res in val_ocr_result:
                                        timeiter, ocr_res = next(iter(ocr_res.items()))
                                        timeiter_int = int(timeiter.split('_')[-1])
                                        
                                        # ------------------ overlay pred ocr ------------------
                                        vis_pred = vis_lq.copy()
                                        vis_polys = ocr_res.polygons.view(-1,16,2)  # b 16 2
                                        vis_recs = ocr_res.recs                     # b 25
                                        for vis_img_idx in range(len(vis_polys)):
                                            pred_poly = vis_polys[vis_img_idx]   # 16 2
                                            pred_poly = np.array(pred_poly.detach().cpu()).astype(np.int32)         
                                            pred_rec = vis_recs[vis_img_idx]     # 25
                                            pred_txt = decode(pred_rec.tolist())
                                            cv2.polylines(vis_pred, [pred_poly], isClosed=True, color=(0,255,0), thickness=2)
                                            cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                        h, w, _ = vis_pred.shape
                                        cv2.putText(vis_pred, str(timeiter_int), (w-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                                        # ------------------ overlay pred ocr ------------------
                                        
                                        vis_result = cv2.hconcat([vis_result, vis_pred])
                                    cv2.imwrite(f'{val_ocr_save_path}/{val_img_id}_step{global_step:09d}.jpg', vis_result[:,:,::-1])
                                    
                                    # save w/ restored results
                                    vis_result = cv2.hconcat([vis_lq, vis_res, vis_hq2, vis_pred, vis_hq1])
                                    cv2.imwrite(f'{val_res_ocr_save_path}/{val_img_id}_step{global_step:09d}.jpg', vis_result[:,:,::-1])
                                                    
                                else:
                                    ## -------------------- visualize only restored results -------------------- 
                                    vis_result = cv2.hconcat([vis_lq, vis_res, vis_hq2])
                                    cv2.imwrite(f'{val_res_ocr_save_path}/{val_img_id}_step{global_step:09d}.jpg', vis_result[:,:,::-1])



            # log 
            logs = {"loss/total_loss": total_loss.detach().item(), 
                    'loss/diff_loss': diff_loss.detach().item(),
                    "optim/lr": lr_scheduler.get_last_lr()[0],
                    }
            
            # ocr log
            if 'ts_module' in cfg.train.model:
                logs["loss/ocr_tot_loss"] = ocr_tot_loss.detach().item()
                logs['optim/ts_module_lr'] = cfg.train.ts_module.lr
                for ocr_key, ocr_val in ocr_loss_dict.items():
                    logs[f"loss/ocr_{ocr_key}"] = ocr_val.detach().item()


            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.train.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.cfg_path = args.config
    if cfg.model.dit.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    main(cfg)
