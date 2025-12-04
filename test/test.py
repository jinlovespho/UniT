
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

import torch
import torch.nn.functional as F 
import torchvision.transforms as T 
from torchvision.utils import save_image

import cv2 
import wandb
import pyiqa
import argparse
import numpy as np 
from PIL import Image 
from omegaconf import OmegaConf

import initialize 
from dataloaders.utils import encode, decode
from pipelines.pipeline_unit import StableDiffusion3ControlNetPipeline

from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix


logger = get_logger(__name__)


def main(cfg):
    
    
    
    # ----------------------------------------
    #         Global reproducability
    # ----------------------------------------
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed)  # seeds Python, NumPy, and PyTorch globally
    
    
    
    
    # ----------------------------------------
    #             Safety check 
    # ----------------------------------------
    val_data_name = cfg.data.val.eval_list[0]
    assert val_data_name in ['realtext', 'satext_lv3', 'satext_lv2', 'satext_lv1', 'satext_test']
    
    
    
    
    # ----------------------------------------
    #           Set experiment name
    # ----------------------------------------
    if cfg.ckpt.resume_path.dit is not None:
        exp_name = cfg.ckpt.resume_path.dit.split('/')[-2]        
    else:
        exp_name = f'dit4sr_baseline'
    exp_name = f'{cfg.data.val.eval_region}__{exp_name}__startpoint-{cfg.data.val.start_point}__alignmethod-{cfg.data.val.align_method}__cfg-{int(cfg.data.val.guidance_scale)}'
    
    if cfg.data.val.text_cond_prompt == 'pred_vlm':
        if cfg.data.val.eval_list[0] == 'realtext':
            vlm_caption_path = cfg.data.val.realtext.vlm_caption_path
        elif cfg.data.val.eval_list[0] == 'satext_lv3':
            vlm_caption_path = cfg.data.val.satext_lv3.vlm_caption_path
        elif cfg.data.val.eval_list[0] == 'satext_lv2':
            vlm_caption_path = cfg.data.val.satext_lv2.vlm_caption_path
        elif cfg.data.val.eval_list[0] == 'satext_lv1':
            vlm_caption_path = cfg.data.val.satext_lv1.vlm_caption_path
        elif cfg.data.val.eval_list[0] == 'satext_test':
            vlm_caption_path = cfg.data.val.satext_test.vlm_caption_path
        cfg.vlm_caption_path = vlm_caption_path
        # english focused input prompt
        question_list = [
            'Please describe the actual objects in the image in a very detailed manner. Please do not include descriptions related to the focus and bokeh of this image. Please do not include descriptions like the background is blurred.',
            "OCR this image and transcribe only the English text.",
            "Read and transcribe all English text visible in this low-resolution image.",
            "Describe the contents of this blurry image, focusing only on any visible English text or characters.",
            "Extract all visible English words and letters from this low-quality image, even if they appear unclear.",
        ]
        exp_name = f'{exp_name}__{cfg.data.val.text_cond_prompt}-prompt__{vlm_caption_path}'
    else:
        exp_name = f'{exp_name}__{cfg.data.val.text_cond_prompt}-prompt'
    
    exp_name = f'{exp_name}__{cfg.log.tracker.msg}'
    exp_name = f'{cfg.log.tracker.msg}'
    cfg.exp_name = exp_name
    print('- EXP NAME: ', exp_name)


    

    # ----------------------------------------
    #         Eval saving directory 
    # ----------------------------------------
    cfg.save.output_dir = f'{cfg.save.output_dir}/{val_data_name}'
    os.makedirs(f'{cfg.save.output_dir}/{exp_name}', exist_ok=True)
    
    
    
    
    # ----------------------------------------
    #       Set accelerator and wandb 
    # ----------------------------------------
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
        wandb.login(key=cfg.log.tracker.key)
        wandb.init(
            project=cfg.log.tracker.project_name,
            name=f'VAL__{cfg.save.output_dir.split("/")[-1]}__{val_data_name}__serv{str(cfg.log.tracker.server)}gpu{str(cfg.log.tracker.gpu)}__{exp_name}',
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    
    
    # -------------------------------------
    #           load val data
    # -------------------------------------
    _, val_datasets = initialize.load_data(cfg)



    # -------------------------------------
    #           load models 
    # -------------------------------------
    models = initialize.load_model(cfg, accelerator)
    


    # -------------------------------------------------
    #   set cuda and proper dtype(float32, float16)
    # -------------------------------------------------
    weight_dtype = initialize.set_model_device(cfg, accelerator, models)



    # -----------------------------------
    #            SR metrics
    # -----------------------------------
    metric_psnr = pyiqa.create_metric('psnr', device=accelerator.device)
    metric_ssim = pyiqa.create_metric('ssimc', device=accelerator.device)
    metric_lpips = pyiqa.create_metric('lpips', device=accelerator.device)
    metric_dists = pyiqa.create_metric('dists', device=accelerator.device)
    metric_niqe = pyiqa.create_metric('niqe', device=accelerator.device)
    metric_musiq = pyiqa.create_metric('musiq', device=accelerator.device)
    metric_maniqa = pyiqa.create_metric('maniqa', device=accelerator.device)
    metric_clipiqa = pyiqa.create_metric('clipiqa', device=accelerator.device)



    # -----------------------------------
    #            load tsm
    # -----------------------------------
    if 'ts_module' in cfg.train.model:
        ts_module = models['testr'] 
    else:
        ts_module = None 
    
    
    
    # -----------------------------------
    #     load validation pipeline
    # -----------------------------------
    val_pipeline = StableDiffusion3ControlNetPipeline(
        vae=models['vae'], text_encoder=models['text_encoders'][0], text_encoder_2=models['text_encoders'][1], text_encoder_3=models['text_encoders'][2], 
        tokenizer=models['tokenizers'][0], tokenizer_2=models['tokenizers'][1], tokenizer_3=models['tokenizers'][2], 
        transformer=models['transformer'], scheduler=models['noise_scheduler'], ts_module=ts_module,
    )
    
    
    if cfg.data.val.vlm.vlm_correction:

        if cfg.data.val.vlm.model == 'qwenvl-3b':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map='auto')
            vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            vlm_model = vlm_model.eval()
        
        elif cfg.data.val.vlm.model == 'qwenvl-7b':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map='auto')
            vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            vlm_model = vlm_model.eval()
    
    
    else:
        vlm_model=None 
        vlm_processor=None
    

    # -----------------------
    #     image metric
    # -----------------------
    metrics={}



    # ------------------------------------------
    #        validation loop (per dataset)
    # ------------------------------------------
    for val_data_name, val_data in val_datasets.items():


        # -------------------------------------------------------------------------
        # Initialize metric for full-image and cropped-image evaluation
        # -------------------------------------------------------------------------

        # ===== Full image metrics =====
        metrics[f'{val_data_name}_full_psnr'] = []
        metrics[f'{val_data_name}_full_ssim'] = []
        metrics[f'{val_data_name}_full_lpips'] = []
        metrics[f'{val_data_name}_full_dists'] = []
        metrics[f'{val_data_name}_full_niqe'] = []
        metrics[f'{val_data_name}_full_musiq'] = []
        metrics[f'{val_data_name}_full_maniqa'] = []
        metrics[f'{val_data_name}_full_clipiqa'] = []

        # Min–max normalized (full)
        metrics[f'{val_data_name}_full_norm_psnr'] = []
        metrics[f'{val_data_name}_full_norm_ssim'] = []
        metrics[f'{val_data_name}_full_norm_lpips'] = []
        metrics[f'{val_data_name}_full_norm_dists'] = []
        metrics[f'{val_data_name}_full_norm_niqe'] = []
        metrics[f'{val_data_name}_full_norm_musiq'] = []
        metrics[f'{val_data_name}_full_norm_maniqa'] = []
        metrics[f'{val_data_name}_full_norm_clipiqa'] = []


        # ===== Cropped region metrics =====
        metrics[f'{val_data_name}_crop_psnr'] = []
        metrics[f'{val_data_name}_crop_ssim'] = []
        metrics[f'{val_data_name}_crop_lpips'] = []
        metrics[f'{val_data_name}_crop_dists'] = []
        metrics[f'{val_data_name}_crop_niqe'] = []
        metrics[f'{val_data_name}_crop_musiq'] = []
        metrics[f'{val_data_name}_crop_maniqa'] = []
        metrics[f'{val_data_name}_crop_clipiqa'] = []

        # Min–max normalized (crop)
        metrics[f'{val_data_name}_crop_norm_psnr'] = []
        metrics[f'{val_data_name}_crop_norm_ssim'] = []
        metrics[f'{val_data_name}_crop_norm_lpips'] = []
        metrics[f'{val_data_name}_crop_norm_dists'] = []
        metrics[f'{val_data_name}_crop_norm_niqe'] = []
        metrics[f'{val_data_name}_crop_norm_musiq'] = []
        metrics[f'{val_data_name}_crop_norm_maniqa'] = []
        metrics[f'{val_data_name}_crop_norm_clipiqa'] = []



        # ------------------------------------------
        #        validation loop (per sample)
        # ------------------------------------------
        for sample_idx, val_sample in enumerate(val_data):
            
            
            # print exp ifo
            print('-------------------------------------------------')
            print(f'{cfg.data.val.eval_region} - {val_data_name} - {sample_idx+1}/{len(val_data)} - using {cfg.data.val.text_cond_prompt}prompt') 
            
            
            # set seed 
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
            
            
            # process hq image 
            val_hq_pil = Image.open(val_hq_path).convert("RGB") 
            
            
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
            #       set input prompt to diffusion model
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
                if cfg.data.val.vlm.vlm_correction:
                    val_init_prompt = [val_vlm_cap]
                else:
                    val_init_prompt = ['']
                
            # vlm prompt 
            elif cfg.data.val.text_cond_prompt == 'pred_vlm':
                val_init_prompt = [val_vlm_cap]
                
            # null prompt 
            elif cfg.data.val.text_cond_prompt == 'null':
                val_init_prompt = ['']
            
            # added prompt             
            if cfg.data.val.added_prompt is not None:
                val_init_prompt = [f'{val_init_prompt[0]} {cfg.data.val.added_prompt}']

            # cfg negative prompt 
            if cfg.data.val.negative_prompt is not None:
                neg_prompt = cfg.data.val.negative_prompt
            else:
                neg_prompt = None 
                
                
            print('INIT PROMPT: ', val_init_prompt)
            # -------------------------------------------------
            #       validation pipeline forward pass 
            # -------------------------------------------------
            with torch.no_grad():
                val_out = val_pipeline(
                    prompt=val_init_prompt[0], control_image=val_lq_pil, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=512, width=512,
                    guidance_scale=cfg.data.val.guidance_scale, negative_prompt=neg_prompt,
                    start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                    output_type = 'pil', return_dict=False, lq_id=val_img_id, val_data_name=val_data_name, cfg=cfg, mode='val', hq_img=val_hq_pil, vlm_model=vlm_model, vlm_processor=vlm_processor, val_lq_path=val_lq_path
                )
            
            
            
            # retrieve restored image 
            val_res_pil = val_out[0][0]   # 1 3 512 512 [0,1]
            
            
            # post processing
            if cfg.data.val.align_method is not None:                
                if cfg.data.val.align_method == 'wavelet':
                    val_res_pil = wavelet_color_fix(val_res_pil, val_lq_pil)
                elif cfg.data.val.align_method == 'adain':
                    val_res_pil = adain_color_fix(val_res_pil, val_lq_pil)
            

            # Save only the restored image
            val_res_save_path = f'{cfg.save.output_dir}/{exp_name}/final_restored_img'
            os.makedirs(val_res_save_path, exist_ok=True)
            val_res_pil.save(f'{val_res_save_path}/{val_img_id}.png')
            

            
            
            # ---------------------------------------
            #       image metric calculation
            # ---------------------------------------
            
            # restored pil img -> tensor 
            val_res_pt = T.ToTensor()(val_res_pil) 
            val_res_pt = val_res_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
            
            # gt pil img -> tensor 
            val_hq_pt = T.ToTensor()(val_hq_pil)
            val_hq_pt = val_hq_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
            
            # lq pil img -> tensor 
            val_lq_pt = T.ToTensor()(val_lq_pil)
            val_lq_pt = val_lq_pt.to(device=accelerator.device, dtype=torch.float32).unsqueeze(dim=0).clamp(0.0, 1.0)   # 1 3 512 512 
            
            
            
    
            # ----------------------------------------------
            #   Eval on full image
            # ----------------------------------------------
            full_metrics = {
                'psnr': torch.mean(metric_psnr(val_res_pt, val_hq_pt)).item(),
                'ssim': torch.mean(metric_ssim(val_res_pt, val_hq_pt)).item(),
                'lpips': torch.mean(metric_lpips(val_res_pt, val_hq_pt)).item(),
                'dists': torch.mean(metric_dists(val_res_pt, val_hq_pt)).item(),
                'niqe': torch.mean(metric_niqe(val_res_pt, val_hq_pt)).item(),
                'musiq': torch.mean(metric_musiq(val_res_pt, val_hq_pt)).item(),
                'maniqa': torch.mean(metric_maniqa(val_res_pt, val_hq_pt)).item(),
                'clipiqa': torch.mean(metric_clipiqa(val_res_pt, val_hq_pt)).item()
            }
            for k, v in full_metrics.items():
                metrics[f'{val_data_name}_full_{k}'].append(v)

            # Min–max normalization
            val_res_pt_norm = (val_res_pt - val_res_pt.min()) / (val_res_pt.max() - val_res_pt.min() + 1e-8)
            val_hq_pt_norm = (val_hq_pt - val_hq_pt.min()) / (val_hq_pt.max() - val_hq_pt.min() + 1e-8)

            full_norm_metrics = {
                'psnr': torch.mean(metric_psnr(val_res_pt_norm, val_hq_pt_norm)).item(),
                'ssim': torch.mean(metric_ssim(val_res_pt_norm, val_hq_pt_norm)).item(),
                'lpips': torch.mean(metric_lpips(val_res_pt_norm, val_hq_pt_norm)).item(),
                'dists': torch.mean(metric_dists(val_res_pt_norm, val_hq_pt_norm)).item(),
                'niqe': torch.mean(metric_niqe(val_res_pt_norm, val_hq_pt_norm)).item(),
                'musiq': torch.mean(metric_musiq(val_res_pt_norm, val_hq_pt_norm)).item(),
                'maniqa': torch.mean(metric_maniqa(val_res_pt_norm, val_hq_pt_norm)).item(),
                'clipiqa': torch.mean(metric_clipiqa(val_res_pt_norm, val_hq_pt_norm)).item()
            }
            for k, v in full_norm_metrics.items():
                metrics[f'{val_data_name}_full_norm_{k}'].append(v)




            # ----------------------------------------------
            #   Eval on bbox cropped text regions
            # ----------------------------------------------
            bbox_metrics = {k: [] for k in ['crop_psnr','crop_ssim','crop_lpips','crop_dists', 'crop_niqe','crop_musiq','crop_maniqa','crop_clipiqa']}
            bbox_metrics_norm = {k: [] for k in ['crop_norm_psnr','crop_norm_ssim','crop_norm_lpips','crop_norm_dists', 'crop_norm_niqe','crop_norm_musiq','crop_norm_maniqa','crop_norm_clipiqa']}

            # Define a dictionary of metric functions
            metric_fn_dict = {
                'psnr': metric_psnr,
                'ssim': metric_ssim,
                'lpips': metric_lpips,
                'dists': metric_dists,
                'niqe': metric_niqe,
                'musiq': metric_musiq,
                'maniqa': metric_maniqa,
                'clipiqa': metric_clipiqa
            }

            MIN_SAFE_SIZE = 96

            for bbox in val_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                res_crop = val_res_pt[:, :, y1:y2, x1:x2]
                hq_crop = val_hq_pt[:, :, y1:y2, x1:x2]

                Hc, Wc = res_crop.shape[-2:]
                if Hc < MIN_SAFE_SIZE or Wc < MIN_SAFE_SIZE:
                    crop_scale = max(MIN_SAFE_SIZE / Hc, MIN_SAFE_SIZE / Wc)
                    new_h, new_w = int(round(Hc * crop_scale)), int(round(Wc * crop_scale))
                    res_crop = F.interpolate(res_crop, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    hq_crop  = F.interpolate(hq_crop, size=(new_h, new_w), mode='bilinear', align_corners=False)

                # Original metrics
                for k in bbox_metrics.keys():
                    metric_name = k.replace('crop_', '')
                    metric_fn = metric_fn_dict[metric_name]
                    bbox_metrics[k].append(torch.mean(metric_fn(res_crop, hq_crop)).item())

                # Normalized metrics
                res_crop_norm = (res_crop - res_crop.min()) / (res_crop.max() - res_crop.min() + 1e-8)
                hq_crop_norm = (hq_crop - hq_crop.min()) / (hq_crop.max() - hq_crop.min() + 1e-8)
                for k in bbox_metrics_norm.keys():
                    metric_name = k.replace('crop_norm_', '')
                    metric_fn = metric_fn_dict[metric_name]
                    bbox_metrics_norm[k].append(torch.mean(metric_fn(res_crop_norm, hq_crop_norm)).item())

            # Average metrics across bounding boxes
            for key, vals in bbox_metrics.items():
                metrics[f'{val_data_name}_{key}'].append(np.mean(vals))
            for key, vals in bbox_metrics_norm.items():
                metrics[f'{val_data_name}_{key}'].append(np.mean(vals))



            # ----------------------------------------------
            # Save image-level metrics to a text file 
            # ----------------------------------------------
            val_img_metric_save_path = f'{cfg.save.output_dir}/{exp_name}/final_img_metric'
            os.makedirs(val_img_metric_save_path, exist_ok=True)
            save_file = f'{val_img_metric_save_path}/{val_img_id}.txt'

            with open(save_file, "w") as f:
                f.write("\n" + "="*100 + "\n")
                f.write(f"Metrics for image: {val_img_id}\n")
                f.write(f"CFG guidance scale: {cfg.data.val.guidance_scale}\n")
                f.write(f"Using {cfg.data.val.text_cond_prompt}\n")
                f.write("="*100 + "\n")
                f.write(f"{'Metric':<10} | {'Full':>10} | {'Full (Norm)':>14} | {'Crop (Avg)':>12} | {'Crop Norm (Avg)':>16}\n")
                f.write("-"*100 + "\n")

                metrics_order = ['psnr','ssim','lpips','dists', 'niqe','musiq','maniqa','clipiqa']
                for m in metrics_order:
                    full_val = full_metrics[m]
                    full_norm_val = full_norm_metrics[m]
                    crop_val = np.mean(bbox_metrics[f'crop_{m}']) if f'crop_{m}' in bbox_metrics else 0.0
                    crop_norm_val = np.mean(bbox_metrics_norm[f'crop_norm_{m}']) if f'crop_norm_{m}' in bbox_metrics_norm else 0.0
                    f.write(f"{m.upper():<10} | {full_val:>10.4f} | {full_norm_val:>14.4f} | {crop_val:>12.4f} | {crop_norm_val:>16.4f}\n")
                f.write("="*100 + "\n")


            
            
            # --------------------------------------------------
            #       restoration and OCR visualization 
            # --------------------------------------------------

            # save path
            val_res_ocr_save_path = f'{cfg.save.output_dir}/{exp_name}/final_result'
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
                val_ocr_save_path = f'{cfg.save.output_dir}/{exp_name}/final_ocr_result'
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
                    cv2.putText(vis_hq1, gt_txt, (gt_poly[0][0], gt_poly[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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
                        cv2.putText(vis_pred, pred_txt, (pred_poly[0][0], pred_poly[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    h, w, _ = vis_pred.shape
                    cv2.putText(vis_pred, str(timeiter_int), (w-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    # ------------------ overlay pred ocr ------------------
                    
                    vis_result = cv2.hconcat([vis_result, vis_pred])
                cv2.imwrite(f'{val_ocr_save_path}/{val_img_id}.jpg', vis_result[:,:,::-1])
                
                # save w/ restored results
                vis_result = cv2.hconcat([vis_lq, vis_res, vis_hq2, vis_pred, vis_hq1])
                cv2.imwrite(f'{val_res_ocr_save_path}/{val_img_id}.jpg', vis_result[:,:,::-1])
                                
            else:
                ## -------------------- visualize only restored results -------------------- 
                vis_result = cv2.hconcat([vis_lq, vis_res, vis_hq2])
                cv2.imwrite(f'{val_res_ocr_save_path}/{val_img_id}.jpg', vis_result[:,:,::-1])



        # --------------------------------------------------
        #       Full image metric calculation
        # --------------------------------------------------
        full_tot_val_psnr = np.array(metrics[f'{val_data_name}_full_psnr']).mean()
        full_tot_val_ssim = np.array(metrics[f'{val_data_name}_full_ssim']).mean()
        full_tot_val_lpips = np.array(metrics[f'{val_data_name}_full_lpips']).mean()
        full_tot_val_dists = np.array(metrics[f'{val_data_name}_full_dists']).mean()
        full_tot_val_niqe = np.array(metrics[f'{val_data_name}_full_niqe']).mean()
        full_tot_val_musiq = np.array(metrics[f'{val_data_name}_full_musiq']).mean()
        full_tot_val_maniqa = np.array(metrics[f'{val_data_name}_full_maniqa']).mean()
        full_tot_val_clipiqa = np.array(metrics[f'{val_data_name}_full_clipiqa']).mean()
        
        # min max normalized metric calculation
        full_norm_tot_val_psnr = np.array(metrics[f'{val_data_name}_full_norm_psnr']).mean()
        full_norm_tot_val_ssim = np.array(metrics[f'{val_data_name}_full_norm_ssim']).mean()
        full_norm_tot_val_lpips = np.array(metrics[f'{val_data_name}_full_norm_lpips']).mean()
        full_norm_tot_val_dists = np.array(metrics[f'{val_data_name}_full_norm_dists']).mean()
        full_norm_tot_val_niqe = np.array(metrics[f'{val_data_name}_full_norm_niqe']).mean()
        full_norm_tot_val_musiq = np.array(metrics[f'{val_data_name}_full_norm_musiq']).mean()
        full_norm_tot_val_maniqa = np.array(metrics[f'{val_data_name}_full_norm_maniqa']).mean()
        full_norm_tot_val_clipiqa = np.array(metrics[f'{val_data_name}_full_norm_clipiqa']).mean()



        # --------------------------------------------------
        #       Cropped image metric calculation
        # --------------------------------------------------
        crop_tot_val_psnr = np.array(metrics[f'{val_data_name}_crop_psnr']).mean()
        crop_tot_val_ssim = np.array(metrics[f'{val_data_name}_crop_ssim']).mean()
        crop_tot_val_lpips = np.array(metrics[f'{val_data_name}_crop_lpips']).mean()
        crop_tot_val_dists = np.array(metrics[f'{val_data_name}_crop_dists']).mean()
        crop_tot_val_niqe = np.array(metrics[f'{val_data_name}_crop_niqe']).mean()
        crop_tot_val_musiq = np.array(metrics[f'{val_data_name}_crop_musiq']).mean()
        crop_tot_val_maniqa = np.array(metrics[f'{val_data_name}_crop_maniqa']).mean()
        crop_tot_val_clipiqa = np.array(metrics[f'{val_data_name}_crop_clipiqa']).mean()
        
        # min max normalized metric calculation
        crop_norm_tot_val_psnr = np.array(metrics[f'{val_data_name}_crop_norm_psnr']).mean()
        crop_norm_tot_val_ssim = np.array(metrics[f'{val_data_name}_crop_norm_ssim']).mean()
        crop_norm_tot_val_lpips = np.array(metrics[f'{val_data_name}_crop_norm_lpips']).mean()
        crop_norm_tot_val_dists = np.array(metrics[f'{val_data_name}_crop_norm_dists']).mean()
        crop_norm_tot_val_niqe = np.array(metrics[f'{val_data_name}_crop_norm_niqe']).mean()
        crop_norm_tot_val_musiq = np.array(metrics[f'{val_data_name}_crop_norm_musiq']).mean()
        crop_norm_tot_val_maniqa = np.array(metrics[f'{val_data_name}_crop_norm_maniqa']).mean()
        crop_norm_tot_val_clipiqa = np.array(metrics[f'{val_data_name}_crop_norm_clipiqa']).mean()


        # -------------------------------------------------------------------------
        # Print and log combined FULL + CROP evaluation results
        # -------------------------------------------------------------------------
        if accelerator.is_main_process:
            output_lines = []
            output_lines.append("\n" + "="*100)
            output_lines.append(f"Validation Results for '{val_data_name}' Dataset")
            output_lines.append("="*100)
            output_lines.append(f"{'Metric':<10} | {'Full':>10} | {'Full (Norm)':>14} | {'Crop':>10} | {'Crop (Norm)':>14}")
            output_lines.append("-"*100)
            output_lines.append(f"{'PSNR':<10} | {full_tot_val_psnr:>10.4f} | {full_norm_tot_val_psnr:>14.4f} | {crop_tot_val_psnr:>10.4f} | {crop_norm_tot_val_psnr:>14.4f}")
            output_lines.append(f"{'SSIM':<10} | {full_tot_val_ssim:>10.4f} | {full_norm_tot_val_ssim:>14.4f} | {crop_tot_val_ssim:>10.4f} | {crop_norm_tot_val_ssim:>14.4f}")
            output_lines.append(f"{'LPIPS':<10} | {full_tot_val_lpips:>10.4f} | {full_norm_tot_val_lpips:>14.4f} | {crop_tot_val_lpips:>10.4f} | {crop_norm_tot_val_lpips:>14.4f}")
            output_lines.append(f"{'DISTS':<10} | {full_tot_val_dists:>10.4f} | {full_norm_tot_val_dists:>14.4f} | {crop_tot_val_dists:>10.4f} | {crop_norm_tot_val_dists:>14.4f}")
            output_lines.append(f"{'NIQE':<10} | {full_tot_val_niqe:>10.4f} | {full_norm_tot_val_niqe:>14.4f} | {crop_tot_val_niqe:>10.4f} | {crop_norm_tot_val_niqe:>14.4f}")
            output_lines.append(f"{'MUSIQ':<10} | {full_tot_val_musiq:>10.4f} | {full_norm_tot_val_musiq:>14.4f} | {crop_tot_val_musiq:>10.4f} | {crop_norm_tot_val_musiq:>14.4f}")
            output_lines.append(f"{'MANIQA':<10} | {full_tot_val_maniqa:>10.4f} | {full_norm_tot_val_maniqa:>14.4f} | {crop_tot_val_maniqa:>10.4f} | {crop_norm_tot_val_maniqa:>14.4f}")
            output_lines.append(f"{'CLIPIQA':<10} | {full_tot_val_clipiqa:>10.4f} | {full_norm_tot_val_clipiqa:>14.4f} | {crop_tot_val_clipiqa:>10.4f} | {crop_norm_tot_val_clipiqa:>14.4f}")
            output_lines.append("="*100 + "\n")

            # Print to console
            for line in output_lines:
                print(line)

            # Save to text file
            val_metric_save_path = f'{cfg.save.output_dir}/{exp_name}'
            os.makedirs(val_metric_save_path, exist_ok=True)
            save_path = f"{val_metric_save_path}/final_{val_data_name}_metric.txt"
            with open(save_path, "w") as f:
                for line in output_lines:
                    f.write(line + "\n")

            # ---------------------------------------------------------------------
            # Log to WandB
            # ---------------------------------------------------------------------
            if cfg.log.tracker.report_to == 'wandb':
                wandb.log({
                    # --- Full Image ---
                    f'val_metric_{val_data_name}_full/psnr': full_tot_val_psnr,
                    f'val_metric_{val_data_name}_full/ssim': full_tot_val_ssim,
                    f'val_metric_{val_data_name}_full/lpips': full_tot_val_lpips,
                    f'val_metric_{val_data_name}_full/dists': full_tot_val_dists,
                    f'val_metric_{val_data_name}_full/niqe': full_tot_val_niqe,
                    f'val_metric_{val_data_name}_full/musiq': full_tot_val_musiq,
                    f'val_metric_{val_data_name}_full/maniqa': full_tot_val_maniqa,
                    f'val_metric_{val_data_name}_full/clipiqa': full_tot_val_clipiqa,

                    f'val_metric_{val_data_name}_full_norm/psnr': full_norm_tot_val_psnr,
                    f'val_metric_{val_data_name}_full_norm/ssim': full_norm_tot_val_ssim,
                    f'val_metric_{val_data_name}_full_norm/lpips': full_norm_tot_val_lpips,
                    f'val_metric_{val_data_name}_full_norm/dists': full_norm_tot_val_dists,
                    f'val_metric_{val_data_name}_full_norm/niqe': full_norm_tot_val_niqe,
                    f'val_metric_{val_data_name}_full_norm/musiq': full_norm_tot_val_musiq,
                    f'val_metric_{val_data_name}_full_norm/maniqa': full_norm_tot_val_maniqa,
                    f'val_metric_{val_data_name}_full_norm/clipiqa': full_norm_tot_val_clipiqa,

                    # --- Cropped Region ---
                    f'val_metric_{val_data_name}_crop/psnr': crop_tot_val_psnr,
                    f'val_metric_{val_data_name}_crop/ssim': crop_tot_val_ssim,
                    f'val_metric_{val_data_name}_crop/lpips': crop_tot_val_lpips,
                    f'val_metric_{val_data_name}_crop/dists': crop_tot_val_dists,
                    f'val_metric_{val_data_name}_crop/niqe': crop_tot_val_niqe,
                    f'val_metric_{val_data_name}_crop/musiq': crop_tot_val_musiq,
                    f'val_metric_{val_data_name}_crop/maniqa': crop_tot_val_maniqa,
                    f'val_metric_{val_data_name}_crop/clipiqa': crop_tot_val_clipiqa,

                    f'val_metric_{val_data_name}_crop_norm/psnr': crop_norm_tot_val_psnr,
                    f'val_metric_{val_data_name}_crop_norm/ssim': crop_norm_tot_val_ssim,
                    f'val_metric_{val_data_name}_crop_norm/lpips': crop_norm_tot_val_lpips,
                    f'val_metric_{val_data_name}_crop_norm/dists': crop_norm_tot_val_dists,
                    f'val_metric_{val_data_name}_crop_norm/niqe': crop_norm_tot_val_niqe,
                    f'val_metric_{val_data_name}_crop_norm/musiq': crop_norm_tot_val_musiq,
                    f'val_metric_{val_data_name}_crop_norm/maniqa': crop_norm_tot_val_maniqa,
                    f'val_metric_{val_data_name}_crop_norm/clipiqa': crop_norm_tot_val_clipiqa,
                })




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
