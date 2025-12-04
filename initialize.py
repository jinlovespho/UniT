import sys
import os 
import cv2
import math
import glob
import yaml
import json
import copy
import shutil
import logging
import argparse
import diffusers
import accelerate
import transformers
from pathlib import Path
from packaging import version
from omegaconf import OmegaConf
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import torch
import torch.utils 
from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset

import numpy as np

import wandb

import bitsandbytes as bnb
from diffusers.training_utils import cast_training_params
from transformers import PretrainedConfig

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

def decode(idxs):
    s = ''
    for idx in idxs:
        if idx < len(CTLABELS):
            s += CTLABELS[idx]
        else:
            return s
    return s


def encode(word):
    s = []
    max_word_len = 25
    for i in range(max_word_len):
        if i < len(word):
            char=word[i]
            idx = CTLABELS.index(char)
            s.append(idx)
        else:
            s.append(96)
    return s


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, cfg):
    text_encoder_one = class_one.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder_2", revision=None, variant=None
    )
    text_encoder_three = class_three.from_pretrained(
        cfg.ckpt.init_path.text_encoder, subfolder="text_encoder_3", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two, text_encoder_three



def load_experiment_setting(cfg, logger, exp_name):
    logging_dir = Path(cfg.save.output_dir, cfg.log.log_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.save.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        log_with=cfg.log.tracker.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.save.output_dir is not None:
            os.makedirs(f'{cfg.save.output_dir}/{exp_name}', exist_ok=True)
            shutil.copyfile(cfg.cfg_path, f'{cfg.save.output_dir}/{exp_name}/train_config.yaml')

            
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    if hasattr(model, "save_pretrained"):  # Hugging Face
                        model.save_pretrained(os.path.join(output_dir, "transformer"))
                    # else:  # Bit of a hacky solution to save non-Hugging Face models - e.g., text spotting module
                    #     ckpt_dict={}
                    #     unwrapped_model = accelerator.unwrap_model(model)
                    #     ckpt_dict['ts_module'] = unwrapped_model.state_dict()
                    #     ckpt_path = f"{output_dir}/ts_module.pt"
                    #     torch.save(ckpt_dict, ckpt_path)
                    i -= 1
        def load_model_hook(models, input_dir):
            breakpoint()
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()
                if hasattr(model, "register_to_config"):
                    load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
                    # load diffusers style into model
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                # else:
                #     ts_ckpt = torch.load(f'{input_dir}/ts_module.pt', map_location="cpu")
                #     load_result = model.load_state_dict(ts_ckpt['ts_module'], strict=False)
                #     print('- TS MODULE load result: ', load_result)
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    return accelerator



def load_trackers(cfg, accelerator, exp_name):
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        wandb.login(key=cfg.log.tracker.key)
        accelerator.init_trackers(
            project_name=cfg.log.tracker.project_name,
            config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True)),
            init_kwargs={
                    'wandb':{
                        'name': f'TRAIN__serv{str(cfg.log.tracker.server)}gpu{str(cfg.log.tracker.gpu)}__{exp_name}',}
                }
        )


def load_val_data(val_data, cfg):
    files=[]
    # load lq, hq
    lq_paths = sorted(glob.glob(f'{val_data.lq_img_path}/*.jpg'))
    hq_paths = sorted(glob.glob(f'{val_data.hq_img_path}/*.jpg'))
    # load ann
    with open(val_data.ann_path, 'r') as f:
        anns = json.load(f)
        anns = sorted(anns.items())
    # for
    val_count=0
    for val_idx, (lq, hq, ann) in enumerate(zip(lq_paths, hq_paths, anns)):
        val_count+=1
        if val_count > val_data.val_num_img:
            break
        # safety check
        lq_id = lq.split('/')[-1].split('.')[0]
        hq_id = hq.split('/')[-1].split('.')[0]
        ann_id = ann[0]
        assert lq_id == hq_id == ann_id

        # eval for specifc ids         
        if (len(val_data.val_img_id) != 0) and (lq_id not in val_data.val_img_id):
            continue 
        
        # load precomputed vlm captions
        # if (cfg.data.val.text_cond_prompt == 'pred_vlm') and (val_data.vlm_caption_path is not None):
        if val_data.vlm_caption_path is not None:
            vlm_captions_txt = sorted(glob.glob(f"{val_data.vlm_caption_path}/*.txt"))
            vlm_caption_txt = vlm_captions_txt[val_idx]
            vlm_cap_id = vlm_caption_txt.split('/')[-1].split('.')[0]
            # safety check
            assert vlm_cap_id == lq_id == hq_id == ann_id
            with open(f'{vlm_caption_txt}', 'r') as f:
                vlm_cap = f.read().strip()
        else:
            vlm_cap = None
            
        
        # process anns
        boxes=[]
        texts=[]
        text_encs=[]
        polys=[]        
        img_anns = ann[1]['0']['text_instances']
        for img_ann in img_anns:
            # process text 
            text = img_ann['text']
            count=0
            for char in text:
                # only allow OCR english vocab: range(32,127)
                if 32 <= ord(char) and ord(char) < 127:
                    count+=1
                    # print(char, ord(char))
            if count == len(text) and count < 26:
                texts.append(text)
                text_encs.append(encode(text))
                assert text == decode(encode(text)), 'check text encoding !'
            else:
                continue
            # process box
            box_xyxy = img_ann['bbox']
            boxes.append(box_xyxy)
            # process polygons
            poly = np.array(img_ann['polygon']).astype(np.int32)    # 16 2
            polys.append(poly)
        # # JLP VIS
        # img0 = cv2.imread(hq)  # 512 512 3
        # img0_box = np.copy(img0)
        # img0_poly = np.copy(img0)
        # x1,y1,x2,y2 = box_xyxy
        # cv2.rectangle(img0_box, (x1,y1), (x2, y2), (0,255,0), 2)
        # cv2.polylines(img0_poly, [poly], True, (0,255,0), 2)
        # cv2.putText(img0_box, text, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # cv2.putText(img0_poly, text, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # cv2.imwrite('./img0_box.jpg', img0_box)
        # cv2.imwrite('./img0_poly.jpg', img0_poly)
        assert len(boxes) == len(texts) == len(text_encs) == len(polys), f" Check loader!"
        # if the filetered image has no bbox and texts, skip it
        if len(boxes) == 0 or len(polys) == 0:
            continue
        
        files.append({  "lq_path": lq,
                        'hq_path': hq,
                        "text": texts, 
                        "bbox": boxes,
                        'poly': polys,
                        'text_enc': text_encs, 
                        "img_id": lq_id,
                        "vlm_cap": vlm_cap})     
    return files



def load_data(cfg):

    # training data
    if cfg.data.train.name == 'satext':
        from basicsr.data.pho_realesrgan_dataset import PhoRealESRGANDataset
        from basicsr.data.pho_realesrgan_dataset import collate_fn_real
        collate_fn = collate_fn_real
        train_ds = PhoRealESRGANDataset(cfg.data.train, mode='train')
        train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, collate_fn=collate_fn)
    else:
        train_loader = None
        
    # validation data
    validation_data={}
    if 'realtext' in cfg.data.val.eval_list:
        validation_data['realtext']=load_val_data(cfg.data.val.realtext, cfg)
    if 'satext_lv3' in cfg.data.val.eval_list:
        validation_data['satext_lv3']=load_val_data(cfg.data.val.satext_lv3, cfg)
    if 'satext_lv2' in cfg.data.val.eval_list:
        validation_data['satext_lv2']=load_val_data(cfg.data.val.satext_lv2, cfg)
    if 'satext_lv1' in cfg.data.val.eval_list:
        validation_data['satext_lv1']=load_val_data(cfg.data.val.satext_lv1, cfg)
    if 'satext_test' in cfg.data.val.eval_list:
        validation_data['satext_test']=load_val_data(cfg.data.val.satext_test, cfg)

    return train_loader, validation_data



def load_model(cfg, accelerator):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

    models = {}

    # load vae 
    vae = AutoencoderKL.from_pretrained(cfg.ckpt.init_path.vae, subfolder="vae", revision=None)
    vae.requires_grad_(False)
    models['vae'] = vae

    # load scheduler 
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg.ckpt.init_path.noise_scheduler, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    models['noise_scheduler'] = noise_scheduler
    models['noise_scheduler_copy'] = noise_scheduler_copy

    # load tokenizer 
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer",
        revision=None,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer_2",
        revision=None,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        cfg.ckpt.init_path.tokenizer,
        subfolder="tokenizer_3",
        revision=None
    )
    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        cfg.ckpt.init_path.text_encoder, None, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, cfg)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    text_encoder_one.eval()
    text_encoder_two.eval()
    text_encoder_three.eval()

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    models['tokenizers'] = tokenizers 
    models['text_encoders'] = text_encoders 
    
    
    # load dit 
    if cfg.ckpt.resume_path.dit is not None:
        dit_ckpt_path = cfg.ckpt.resume_path.dit
    else: 
        dit_ckpt_path = cfg.ckpt.init_path.dit
    
    if cfg.train.transformer.architecture == 'dit4sr':
        from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
        transformer = SD3Transformer2DModel.from_pretrained_local(dit_ckpt_path, subfolder="transformer", revision=None, variant=None, accelerator=accelerator, cfg=cfg)
        if accelerator.is_main_process:
            print('-'*50)
            print('     Using dit4sr architecture ')
    
            
    elif cfg.train.transformer.architecture == 'dit4sr_ocrbranch_ocr2hq':
        from model_dit4sr.transformer_sd3_ocrbranch_ocr2hq import SD3Transformer2DModel_OCRBranch_OCR2HQ
        transformer = SD3Transformer2DModel_OCRBranch_OCR2HQ.from_pretrained_local(dit_ckpt_path, subfolder="transformer", revision=None, variant=None, accelerator=accelerator, cfg=cfg)
        if accelerator.is_main_process:
            print('-'*50)
            print('     Using dit4sr_ocrbranch_OCR2HQ architecture')
    
    
    elif cfg.train.transformer.architecture == 'dit4sr_ocrbranch_ocr2hq2ocr':
        from model_dit4sr.transformer_sd3_ocrbranch_ocr2hq2ocr import SD3Transformer2DModel_OCRBranch_OCR2HQ2OCR
        transformer = SD3Transformer2DModel_OCRBranch_OCR2HQ2OCR.from_pretrained_local(dit_ckpt_path, subfolder="transformer", revision=None, variant=None, accelerator=accelerator, cfg=cfg)
        if accelerator.is_main_process:
            print('-'*50)
            print('     Using dit4sr_ocrbranch_OCR2HQ2OCR architecture')

            
    transformer.requires_grad_(False)
    models['transformer'] = transformer
    if accelerator.is_main_process:
        print(f"-- DiT4SR Checkpoint: {dit_ckpt_path} --")
        print('-'*50)
    
    
    # load ts module 
    if 'ts_module' in cfg.train.model:
        
        if cfg.train.ts_module.architecture == 'testr':
            from testr.adet.modeling.transformer_detector import TransformerDetector
            from testr.adet.config import get_cfg
            # get testr config
            config_testr = get_cfg()
            config_testr.merge_from_file('./testr/configs/TESTR/TESTR_R_50_Polygon.yaml')
            config_testr.diff_feat_extract = cfg.train.transformer.feat_extract
            config_testr.freeze()
            # load testr model
            detector = TransformerDetector(config_testr)
            # load testr pretrained weights     
            if cfg.ckpt.resume_path.ts_module is not None:
                # tsm_ckpt_id = int(cfg.ckpt.resume_path.ts_module.split('/')[-1].split('-')[-1])
                # tsm_ckpt_path = f"{cfg.ckpt.resume_path.ts_module}/ts_module{tsm_ckpt_id:07d}.pt"
                tsm_ckpt_path = cfg.ckpt.resume_path.ts_module
                ckpt = torch.load(tsm_ckpt_path, map_location="cpu")
                load_result = detector.load_state_dict(ckpt["ts_module"], strict=False)

                if accelerator.is_main_process:
                    print("\n──────────────────────────────")
                    print(" [TESTR] Resumed from checkpoint")
                    print(f"  Path: {tsm_ckpt_path}")
                    print(f"  Missing Keys: {load_result.missing_keys}")
                    # print(f"  Unexpected Keys: {load_result.unexpected_keys}")
                    print("──────────────────────────────\n")

            else:
                if cfg.ckpt.init_path.ts_module is not None:
                    tsm_ckpt_path = cfg.ckpt.init_path.ts_module
                    ckpt = torch.load(tsm_ckpt_path, map_location="cpu")
                    load_result = detector.load_state_dict(ckpt["model"], strict=False)

                    if accelerator.is_main_process:
                        print("\n──────────────────────────────")
                        print(" [TESTR] Initialized from checkpoint")
                        print(f"  Path: {tsm_ckpt_path}")
                        print(f"  Missing Keys: {load_result.missing_keys}")
                        # print(f"  Unexpected Keys: {load_result.unexpected_keys}")
                        print("──────────────────────────────\n")
                else:
                    if accelerator.is_main_process:
                        print("\n──────────────────────────────")
                        print(" [TESTR] Initialized from scratch")
                        print("──────────────────────────────\n")

            models["testr"] = detector.train()


    # # load vlm captioner 
    # if not cfg.model.dit.load_precomputed_caption:
    #     from llava.llm_agent import LLavaAgent
    #     from CKPT_PTH import LLAVA_MODEL_PATH
    #     llava_agent = LLavaAgent(LLAVA_MODEL_PATH, accelerator.device, load_8bit=True, load_4bit=False)
    #     models['vlm_agent'] = llava_agent

    return models 


def load_model_params(cfg, accelerator, models):

    tot_param_names=[]
    train_param_names=[]
    frozen_param_names=[]

    tot_param_count=0
    train_param_count=0
    frozen_param_count=0
    
    # # load image restoration module 
    # if 'dit4sr' in cfg.train.model:
    #     for name, param in models['transformer'].named_parameters():
    #         numel = param.numel()
    #         tot_param_count += numel
    #         tot_param_names.append(name)

    #         train_this_param = False
    #         # train lr branch
    #         if ('lrbranch' in cfg.train.finetune) and ('control' in name):
    #             train_this_param = True 
    #         # train all attention layers
    #         if ('attns' in cfg.train.finetune) and ('attn' in name):
    #             train_this_param = True 
            
    #         if train_this_param:
    #             param.requires_grad = True
    #             train_param_count += numel
    #             train_param_names.append(name)
    #         else:
    #             param.requires_grad = False
    #             frozen_param_count += numel
    #             frozen_param_names.append(name)
    
    
    
    # set trainable params in transformer
    if 'transformer' in cfg.train.model:
        for layer_full_name, param in models['transformer'].named_parameters():

            numel = param.numel()
            tot_param_count += numel
            tot_param_names.append(layer_full_name)

            
            # default is set to False
            train_this_param = False
            
            
            # loop through layers to find finetune layers
            layers = layer_full_name.split('.')
            ft_layer_names = cfg.train.transformer.finetune_layer_names
            for layer in layers:
                if layer in ft_layer_names:
                    train_this_param = True 
                    break
            
            
            if train_this_param:
                param.requires_grad = True
                train_param_count += numel
                train_param_names.append(layer_full_name)
            else:
                param.requires_grad = False
                frozen_param_count += numel
                frozen_param_names.append(layer_full_name)
    
    

        
    # set trainable params in text spotting module 
    if 'ts_module' in cfg.train.model:
        if cfg.train.ts_module.architecture == 'testr':
            if len(cfg.train.ts_module.finetune_layer_names) == 0:
                for name, param in models['testr'].named_parameters():
                    # Count total parameters
                    numel = param.numel()
                    tot_param_count += numel
                    tot_param_names.append(f"testr.{name}")

                    # Enable training
                    param.requires_grad = True
                    train_param_count += numel
                    train_param_names.append(f"testr.{name}")
            else:
                raise ValueError()
        else:
            raise ValueError()


    model_params = {}
    model_params['tot_param_names'] = tot_param_names
    model_params['train_param_names'] = train_param_names
    model_params['frozen_param_names'] = frozen_param_names
    model_params['tot_param'] = tot_param_count
    model_params['train_param'] = train_param_count
    model_params['frozen_param'] = frozen_param_count
    
    return model_params




def load_optim(cfg, accelerator, models):

    # # scale lr
    # if cfg.train.scale_lr:
    #     cfg.train.learning_rate.dit = (cfg.train.learning_rate.dit * cfg.train.gradient_accumulation_steps * cfg.train.batch_size * accelerator.num_processes)

    # setup optimizer class
    optimizer_class = bnb.optim.AdamW8bit if cfg.train.use_8bit_adam else torch.optim.AdamW
    
    param_groups=[]
    if 'transformer' in cfg.train.model:
        transformer_params = list(filter(lambda p: p.requires_grad, models['transformer'].parameters()))
        param_groups.append({"params": transformer_params, "lr": cfg.train.transformer.lr})

    if 'ts_module' in cfg.train.model:
        testr_params = list(filter(lambda p: p.requires_grad, models['testr'].parameters()))
        param_groups.append({"params": testr_params, "lr": cfg.train.ts_module.lr})


    # optimizer
    optimizer = optimizer_class(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    return optimizer



def set_model_device(cfg, accelerator, models):

    # Choose dtype for inference-only models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move models to device
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            models[name] = model.to(accelerator.device, dtype=weight_dtype)
        elif isinstance(model, list) and name == 'text_encoders':
            models[name] = [mod.to(accelerator.device, dtype=weight_dtype) for mod in model]
        else:
            # leave schedulers, tokenizers, etc. as they are
            pass


    # Ensure trainable params are in fp32 (LoRA or finetuning)
    if cfg.train.mixed_precision == "fp16":
        if 'ts_module' in cfg.train.model:
            fp32_models = [models['transformer'], models['testr']]
        else:
            fp32_models = [models['transformer']]
        cast_training_params(fp32_models, dtype=torch.float32)
        
    
    # As testr's deformable attention is does not support automatic float32 conversion, it cannot be wrapped around the accelerator.prepare(), which handles ddp training
    # thus testr must be wrapped manually here 
    if torch.distributed.is_initialized() and torch.cuda.device_count() > 1:
        models['testr'] = torch.nn.parallel.DistributedDataParallel(
            models['testr'],
            device_ids=[accelerator.device],
            output_device=accelerator.device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
    

    '''
        for accelerator.mixed_precision == "fp16", we first put all models on fp16
        and then only for parameters that have requires_grad=True, which are trainable parameters,
        they are only set to fp32 by cast_training_params function above.
    '''
    
    
    return weight_dtype
