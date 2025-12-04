# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os 
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
# from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
# from diffusers.models.transformers import SD3Transformer2DModel
# from model_SD3.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from model_unit.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from utils.vaehook import VAEHook

from einops import rearrange 
from torch.cuda.amp import autocast
from dataloaders.utils import encode, decode 
import numpy as np 
from train.train_utils import encode_prompt

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from qwen_vl_utils import process_vision_info

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3ControlNetPipeline
        >>> from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
        >>> from diffusers.utils import load_image

        >>> controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)

        >>> pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
        >>> prompt = "A girl holding a sign that says InstantX"
        >>> image = pipe(prompt, control_image=control_image, controlnet_conditioning_scale=0.7).images[0]
        >>> image.save("sd3.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion3ControlNetPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        controlnet ([`SD3ControlNetModel`] or `List[SD3ControlNetModel]` or [`SD3MultiControlNetModel`]):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        tokenizer_2: CLIPTokenizer,
        ts_module = None,
    ):
        super().__init__()

        if ts_module is not None:
            self.ts_module = ts_module

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)


    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            # breakpoint()
            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(           # clip-L: 1 77 768, 1 768
                prompt=prompt,      
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(       # clip-G: 1 77 1280, 1 1280
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)      # 1 77 2048 -> [clip-L, clip-G] concat channel dim

            t5_prompt_embed = self._get_t5_prompt_embeds(                               # t5: 1 256 4096
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(                               # 1 77 2048+2048(zeropad)
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)    # concat clip and t5 tkns -> tkn dim-> 1 77+256 4096
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")


        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, 16, 1, 1))

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        control_image: PipelineImageInput = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        start_point = 'noise',
        latent_tiled_size=320,
        latent_tiled_overlap=4,
        args=None,
        cfg=None,
        mode=None,
        vlm_model=None,
        vlm_processor=None,
        val_lq_path=None,
        **kwargs
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            controlnet_pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of controlnet input conditions.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype


        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )


        control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,
            )
        control_image_pt = control_image.clone().detach()
        # control_image = (control_image + 1.0) / 2.0
        height, width = control_image.shape[-2:]

        # image_embedding = torch.nn.functional.interpolate(control_image, (512, 512))
        # image_embedding = self.vae.encode(image_embedding).latent_dist.sample()
        # image_embedding = image_embedding * self.vae.config.scaling_factor

        # image_embedding = image_embedding.view(image_embedding.shape[0], 16, -1)
        # # pad_tensor = torch.zeros(control_image.shape[0], 77 - image_embedding.shape[1], 4096).to(image_embedding.device, dtype=dtype)
        # # image_embedding = torch.cat([image_embedding, pad_tensor], dim=1)
        # prompt_embeds = torch.cat([prompt_embeds, image_embedding], dim=-2)


        control_image = self.vae.encode(control_image).latent_dist.sample() # b 3 512 512 
        control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # num_channels_latents = self.transformer.config.in_channels
        num_channels_latents = 16
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # control_image = torch.cat([latents, control_image], dim=0)
        # 6. Prepare the start point
        if start_point == 'noise':
            latents = latents
        elif start_point == 'lr': # LRE Strategy
            # latents_condition_image = self.vae.encode(control_image*2-1).latent_dist.sample()
            # latents_condition_image = latents_condition_image * self.vae.config.scaling_factor
            latents_condition_image = control_image[:1]
            sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
            sigma = sigmas[0].flatten()
            while len(sigma.shape) < 4:
                sigma = sigma.unsqueeze(-1)
            latents = (1.0 - sigma) * latents_condition_image + sigma * latents

        train_glob_step = kwargs.get('global_step')

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            _, _, h, w = latents.size()
            tile_size, tile_overlap = (latent_tiled_size, latent_tiled_overlap) if args is not None else (256, 8)
            if h*w<=tile_size*tile_size:
                print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            else:
                print(f"[Tiled Latent]: the input size is {latents.shape[-2]}x{latents.shape[-1]}, need to tiled")


            # save prompt
            if cfg.data.val.save_prompts:
                # set save path for train/val
                if mode == 'train':
                    txt_save_path = f"{cfg.save.output_dir}/{cfg.exp_name}/{kwargs['val_data_name']}/final_pred_txt"
                elif mode =='val':
                    txt_save_path = f"{cfg.save.output_dir}/{cfg.exp_name}/final_pred_txt"
                os.makedirs(txt_save_path, exist_ok=True)
                
                # set save name for train/val
                if train_glob_step is not None:
                    txt_file = f"{txt_save_path}/{kwargs['lq_id']}_step{train_glob_step:09d}.txt"
                else:
                    txt_file = f"{txt_save_path}/{kwargs['lq_id']}.txt"
                    
                # save txt file content
                with open(txt_file, "w") as f:
                    f.write(f"{kwargs['lq_id']}\n")
                    f.write(f'[text_cond_prompt]: {cfg.data.val.text_cond_prompt}\n')
                    if cfg.data.val.text_cond_prompt == 'pred_vlm':
                        f.write(f'[vlm caption path]: {cfg.vlm_caption_path}\n')
                        # f.write(f'[vlm captioner]: {cfg.vlm_captioner}\n')
                        # f.write(f'[vlm input ques {cfg.vlm_input_ques_num}]: {cfg.vlm_input_ques}\n')
                    f.write(f'[text cond prompt style]: {cfg.model.dit.text_condition.caption_style}\n')
                    f.write(f'[init prompt]: {prompt}\n\n')


            print('use_cfg: ', self.do_classifier_free_guidance)
            val_ocr_result=[]
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # latent_model_input = torch.cat([latents, control_image], dim=1)
                latent_model_input = latents    # b 16 64 64

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latent_model_input] * 2) if self.do_classifier_free_guidance else latent_model_input
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if h*w<=tile_size*tile_size: # tiled latent input
                    
                    # TEXTUAL PROMPT GUIDANCE (TSM) + VLM CORRECTION
                    if (i>0):
                        
                        if ((cfg.data.val.text_cond_prompt == 'pred_tsm') and ('ts_module' in cfg.train.model)):
                            prompt_embeds_input = prompt_embeds_tsm 
                        
                    else:
                        prompt_embeds_input = prompt_embeds
                        
                    
                    if negative_prompt_embeds is not None:
                        negative_prompt_embeds_input = negative_prompt_embeds
                        
                        
                    if self.do_classifier_free_guidance:
                        prompt_embeds_input = torch.cat([negative_prompt_embeds_input, prompt_embeds_input], dim=0)             # (2, 333, 4096)
                        pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)    # (2, 2048)
                    else:
                        # # TEXTUAL PROMPT GUIDANCE (TSM)
                        if (i>0):
                            if ((cfg.data.val.text_cond_prompt == 'pred_tsm') and ('ts_module' in cfg.train.model)):
                                pooled_prompt_embeds_input = pooled_prompt_embeds_tsm
                        else:
                            pooled_prompt_embeds_input = pooled_prompt_embeds
                    
                    # breakpoint()
                    # controlnet(s) inference
                    trans_out = self.transformer(
                        hidden_states=latent_model_input,                       # b 16 64 64 
                        controlnet_image=control_image,                         # b 16 64 64 
                        timestep=timestep,                                      # b      
                        encoder_hidden_states=prompt_embeds_input,              # b 333 4096
                        pooled_projections=pooled_prompt_embeds_input,          # b 2048
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        cfg=cfg
                    )
                    noise_pred = trans_out[0]
                    
                    
                    if cfg.data.val.attn.vis_map:
                        # ------------------------------------
                        #            VIS ATTN MAP 
                        # ------------------------------------
                        
                        attn_maps=[]
                        for trans_blk in self.transformer.transformer_blocks:
                            attn_map = trans_blk.attn.processor.attn_map
                            attn_maps.append(attn_map)  # 2 24 2381 2381
                        

                        
                        num_img_tkn = 1024
                        num_txt_tkn = 333 
                        
                        hq_idx = num_img_tkn
                        lq_idx = num_img_tkn + num_img_tkn
                        txt_idx = num_img_tkn + num_img_tkn + num_txt_tkn

                        dict_maps = {k: [] for k in ['h2h','h2l','h2t','l2h','l2l','l2t','t2h','t2l','t2t']}

                        for m in attn_maps:
                            dict_maps['h2h'].append(m[:, :, 0:hq_idx, 0:hq_idx])
                            dict_maps['h2l'].append(m[:, :, 0:hq_idx, hq_idx:lq_idx])
                            dict_maps['h2t'].append(m[:, :, 0:hq_idx, lq_idx:txt_idx])
                            dict_maps['l2h'].append(m[:, :, hq_idx:lq_idx, 0:hq_idx])
                            dict_maps['l2l'].append(m[:, :, hq_idx:lq_idx, hq_idx:lq_idx])
                            dict_maps['l2t'].append(m[:, :, hq_idx:lq_idx, lq_idx:txt_idx])
                            dict_maps['t2h'].append(m[:, :, lq_idx:txt_idx, 0:hq_idx])
                            dict_maps['t2l'].append(m[:, :, lq_idx:txt_idx, hq_idx:lq_idx])
                            dict_maps['t2t'].append(m[:, :, lq_idx:txt_idx, lq_idx:txt_idx])
                        
                        
                        
                        # ------------- prepare img and prompt -------------
                        lq_img = control_image_pt 
                        hq_img = kwargs['hq_img']
                        
                        clip_prompt_tkn = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids
                        t5_prompt_tkn = self.tokenizer_3(prompt, padding='max_length', max_length=256, truncation=True, add_special_tokens=True, return_tensors='pt').input_ids
                        
                        clip_decode_prompt = self.tokenizer.batch_decode(clip_prompt_tkn)
                        t5_decode_prompt = self.tokenizer_3.batch_decode(t5_prompt_tkn)
                        
                        
                        # ------------- visualize h2t -------------
                        maps = torch.stack(dict_maps['h2t'])    # 24 2 24 1024 333 (layer cfg head dim dim)
                        pos_maps = maps[:,-1]                   # 24 24 1024 333, get cfg pos map [neg, pos]
                        neg_maps = maps[:, 0]
                        map = pos_maps.mean(dim=(0,1))              # 1024 333, avg layer and head


                        map = map.transpose(0,1)
                        map_clip = map[0:77] 
                        map_t5 = map[77:]
                        

                        
                        breakpoint()
                        
                        
                        # ------------- visualize t2h -------------
                        lq_img = control_image_pt
                        prompt = prompt 
                        maps = torch.stack(dict_maps['t2h'])    # 24 2 24 333 1024 (layer cfg head dim dim)
                        pos_maps = maps[:,-1]                   # 24 24 333 1024, get cfg pos map [neg, pos]
                        neg_maps = maps[:, 0]
                        map = pos_maps.mean(dim=(0,1))              # 333 1024, avg layer and head



                    
                    
                    
                    
                    
                    
    
                    # ts module forward pass 
                    if 'ts_module' in cfg.train.model:
                        if len(trans_out) > 1:
                            etc_out = trans_out[1]
                            # unpatchify
                            patch_size = self.transformer.config.patch_size  # 2
                            hidden_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim     # 1536
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
                    
                            
                            
                            extracted_feats = [f.to(torch.float32) for f in extracted_feats]
                            with torch.cuda.amp.autocast(enabled=False):
                                with torch.no_grad():
                                    _, ocr_result = self.ts_module(extracted_feats, targets=None, MODE='VAL')                        
                            results_per_img = ocr_result[0]
                            
                            
                            
                            # save ocr results for ocr visualization
                            if cfg.data.val.ocr.vis_ocr:
                                # visualize all timesteps
                                if -1 in cfg.data.val.ocr.vis_timestep:
                                    val_ocr_result.append({f'timeiter_{i}': results_per_img})
                                # visualize selected timesteps
                                elif i in cfg.data.val.ocr.vis_timestep:
                                    val_ocr_result.append({f'timeiter_{i}': results_per_img})
                                    


                            ts_pred_text=[]
                            # pred_polys=[]
                            
                            for j in range(len(results_per_img.polygons)):
                                val_ctrl_pnt= results_per_img.polygons[j].view(16,2).cpu().detach().numpy().astype(np.int32)    # 32 -> 16 2
                                val_rec = results_per_img.recs[j]
                                val_pred_text = decode(val_rec)
                                
                                # pred_polys.append(val_ctrl_pnt)
                                ts_pred_text.append(val_pred_text)
                            

                            #  set prompt style 
                            texts = [f'"{t}"' for t in ts_pred_text]
                            if cfg.model.dit.text_condition.caption_style == 'descriptive':
                                tsm_pred_prompt = [f'The image features the texts {", ".join(texts)} that appear clearly on signs, boards, buildings, or other objects.']
                                if len(tsm_pred_prompt) == 0:
                                    tsm_pred_prompt=[""]
                            elif cfg.model.dit.text_condition.caption_style == 'tag':
                                tsm_pred_prompt = [f"{', '.join(texts)}"]
                            
                            
                            
                            
                            # -------------------------------------------------
                            #                   vlm correction
                            # -------------------------------------------------    
                            if cfg.data.val.vlm.vlm_correction:
                                
                                if i < min(cfg.data.val.vlm.vlm_apply_at_iter):
                                    # print(f'using TSM prompt - iter{i}')
                                    pred_prompt = prompt
                                

                                elif i in cfg.data.val.vlm.vlm_apply_at_iter:
                                    
                                    # Build instruction message
                                    if len(texts) == 0:
                                        predicted_texts_str = "No predicted texts are available."
                                        hint_block = ""
                                    else:
                                        predicted_texts_str = ", ".join(texts)
                                        hint_block = f"Use the following predicted texts only as hints: {predicted_texts_str}."

                                    vlm_instruction = (
                                        "You are given a low-quality image containing degraded English text. "
                                        f"{hint_block} "
                                        "Your task is to recover the correct text from the image.\n\n"
                                        "Instructions:\n"
                                        "1. Look carefully at the image to infer the actual text.\n"
                                        "2. Use predicted texts only as supportive clues.\n"
                                        "3. Correct OCR errors, noise, or missing characters.\n"
                                        "4. Do NOT hallucinate text that is not visible.\n"
                                        "5. Output only the corrected text as a clean list."
                                    )
                                    
                                    
                                    messages = [
                                                    {
                                                        "role": "user",
                                                        "content": [
                                                            {
                                                                "type": "image",
                                                                "image": val_lq_path,
                                                            },
                                                            {"type": "text", 
                                                            "text": vlm_instruction}
                                                        ],
                                                    }
                                                ]



                                    # Preparation for inference
                                    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                    image_inputs, video_inputs = process_vision_info(messages)
                                    inputs = vlm_processor(
                                        text=[text],
                                        images=image_inputs,
                                        videos=video_inputs,
                                        padding=True,
                                        return_tensors="pt",
                                    )
                                    inputs = inputs.to(vlm_model.device)
                                    # Inference: Generation of the output
                                    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
                                    generated_ids_trimmed = [
                                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                    ]
                                    vlm_pred_texts = vlm_processor.batch_decode(
                                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                    )
                                    
                                    
                                    if type(vlm_pred_texts) == list and len(vlm_pred_texts) != 0:
                                        
                                        # remove unwanted characters 
                                        vlm_clean_texts=[]
                                        for char in vlm_pred_texts[0]:
                                            if 32 <= ord(char) and ord(char) < 127:
                                                vlm_clean_texts.append(char)
                                        vlm_pred_texts=''.join(vlm_clean_texts)
                                        
                                        # additional filtering
                                        vlm_pred_texts = vlm_pred_texts.replace('[', '')
                                        vlm_pred_texts = vlm_pred_texts.replace(']', '')
                                        vlm_pred_texts = vlm_pred_texts.replace("'", "")
                                        vlm_pred_texts = vlm_pred_texts.replace("-", "")
                                        
                                    # print('iter: ', i)
                                    # print('tsm text: ', predicted_texts_str)
                                    # print('vlm correction: ', vlm_pred_texts)
                                    # print(f'using vlm prompt - iter{i}')
                                    pred_prompt = [vlm_pred_texts]
                                

                                else:
                                    # print(f'using previous vlm prompt - iter{i}')
                                    pred_prompt = [vlm_pred_texts]
                                    
                            # not using vlm correction -> just use tsm prompt
                            else:
                                pred_prompt = tsm_pred_prompt
                                
                            
                            # added prompt             
                            if cfg.data.val.added_prompt is not None:
                                pred_prompt = [f'{pred_prompt[0]} {cfg.data.val.added_prompt}']
                                
                            
                            # print and save prompt
                            if cfg.data.val.save_prompts:
                                
                                if cfg.data.val.text_cond_prompt == 'pred_tsm':    
                                    
                                    if cfg.data.val.vlm.vlm_correction:
                                        
                                        if i < min(cfg.data.val.vlm.vlm_apply_at_iter):
                                            print(f"iter: {i:02d} | timestep: {t.item():8.2f} | current tsm ocr: {ts_pred_text}")
                                            with open(txt_file, "a") as f:
                                                f.write(f"iter: {i:02d}   |   timestep: {t.item():8.2f}   |   current tsm ocr: {ts_pred_text}\n")
                                        
                                        elif i in cfg.data.val.vlm.vlm_apply_at_iter:
                                            print(f"iter: {i:02d} | timestep: {t.item():8.2f} | APPLY VLM CORRECTION: [{vlm_pred_texts}]")
                                            with open(txt_file, "a") as f:
                                                f.write(f"iter: {i:02d}   |   timestep: {t.item():8.2f}   |   APPLY VLM CORRECTION: [{vlm_pred_texts}]\n")
                                        
                                        else:
                                            print(f"iter: {i:02d} | timestep: {t.item():8.2f} | vlm_corrected prompt: [{vlm_pred_texts}]")
                                            with open(txt_file, "a") as f:
                                                f.write(f"iter: {i:02d}   |   timestep: {t.item():8.2f}   |   vlm_corrected prompt: [{vlm_pred_texts}]\n")
                                            
                                    else:
                                        print(f"iter: {i:02d} | timestep: {t.item():8.2f} | text prompt: {ts_pred_text}")
                                        with open(txt_file, "a") as f:
                                            f.write(f"iter: {i:02d}   |   timestep: {t.item():8.2f}   |   text prompt: {ts_pred_text}\n")
                                
                                else:
                                    print(f"iter: {i:02d} | timestep: {t.item():8.2f} | text prompt: {prompt}")
                                    with open(txt_file, "a") as f:
                                        f.write(f"iter: {i:02d}   |   timestep: {t.item():8.2f}   |   text prompt: {prompt}\n")
                                    
                            
                            # ts module inference prompt 
                            (
                                prompt_embeds_tsm,
                                _,
                                pooled_prompt_embeds_tsm,
                                _,
                            ) = self.encode_prompt(
                                prompt=pred_prompt,
                                prompt_2=None,
                                prompt_3=None,
                                negative_prompt=None,
                                negative_prompt_2=None,
                                negative_prompt_3=None,
                                do_classifier_free_guidance=False,
                                prompt_embeds=None,
                                negative_prompt_embeds=None,
                                pooled_prompt_embeds=None,
                                negative_pooled_prompt_embeds=None,
                                device=device,
                                clip_skip=None,
                                num_images_per_prompt=num_images_per_prompt,
                                max_sequence_length=max_sequence_length,
                            )
                            

                else:
                    tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
                    tile_size = min(tile_size, min(h, w))
                    tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

                    grid_rows = 0
                    cur_x = 0
                    while cur_x < latent_model_input.size(-1):
                        cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                        grid_rows += 1

                    grid_cols = 0
                    cur_y = 0
                    while cur_y < latent_model_input.size(-2):
                        cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                        grid_cols += 1

                    input_list = []
                    cond_list = []
                    img_list = []
                    noise_preds = []
                    for row in range(grid_rows):
                        noise_preds_row = []
                        for col in range(grid_cols):
                            if col < grid_cols-1 or row < grid_rows-1:
                                # extract tile from input image
                                ofs_x = max(row * tile_size-tile_overlap * row, 0)
                                ofs_y = max(col * tile_size-tile_overlap * col, 0)
                                # input tile area on total image
                            if row == grid_rows-1:
                                ofs_x = w - tile_size
                            if col == grid_cols-1:
                                ofs_y = h - tile_size

                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size

                            # input tile dimensions
                            input_tile = latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            input_list.append(input_tile)
                            cond_tile = control_image[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            cond_list.append(cond_tile)
                            # img_tile = image[:, :, input_start_y*8:input_end_y*8, input_start_x*8:input_end_x*8]
                            # img_list.append(img_tile)

                            if len(input_list) == batch_size or col == grid_cols-1:
                                input_list_t = torch.cat(input_list, dim=0)
                                cond_list_t = torch.cat(cond_list, dim=0)

                                # image_embedding = cond_list_t.view(cond_list_t.shape[0], 16, -1)
                                # prompt_embeds_input = torch.cat([prompt_embeds, image_embedding], dim=-2)
                                prompt_embeds_input = prompt_embeds
                                if negative_prompt_embeds is not None:
                                    # negative_prompt_embeds_input = torch.cat([negative_prompt_embeds, image_embedding], dim=-2)
                                    negative_prompt_embeds_input = negative_prompt_embeds

                                if self.do_classifier_free_guidance:
                                    prompt_embeds_input = torch.cat([negative_prompt_embeds_input, prompt_embeds_input], dim=0)
                                    pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                                else:
                                    pooled_prompt_embeds_input = pooled_prompt_embeds

                                # img_list_t = torch.cat(img_list, dim=0)
                                #print(input_list_t.shape, cond_list_t.shape, img_list_t.shape, fg_mask_list_t.shape)

                                noise_pred = self.transformer(
                                    hidden_states=input_list_t,
                                    controlnet_image=cond_list_t,
                                    timestep=timestep,
                                    encoder_hidden_states=prompt_embeds_input,
                                    pooled_projections=pooled_prompt_embeds_input,
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]

                                #for sample_i in range(model_out.size(0)):
                                #    noise_preds_row.append(model_out[sample_i].unsqueeze(0))
                                input_list = []
                                cond_list = []
                                img_list = []

                            noise_preds.append(noise_pred)

                    # Stitch noise predictions for all tiles
                    noise_pred = torch.zeros(latent_model_input.shape, device=latents.device)
                    contributors = torch.zeros(latent_model_input.shape, device=latents.device)
                    # Add each tile contribution to overall latents
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if col < grid_cols-1 or row < grid_rows-1:
                                # extract tile from input image
                                ofs_x = max(row * tile_size-tile_overlap * row, 0)
                                ofs_y = max(col * tile_size-tile_overlap * col, 0)
                                # input tile area on total image
                            if row == grid_rows-1:
                                ofs_x = w - tile_size
                            if col == grid_cols-1:
                                ofs_y = h - tile_size

                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size
    
                            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                    # Average overlapping areas with more than 1 contributor
                    noise_pred /= contributors

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]


                # if latents.dtype != latents_dtype:
                #     print('TRUE')
                #     if torch.backends.mps.is_available():
                #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                #         latents = latents.to(latents_dtype)

                # if callback_on_step_end is not None:
                #     breakpoint()
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                #     negative_pooled_prompt_embeds = callback_outputs.pop(
                #         "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                #     )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()
            

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            with torch.cuda.amp.autocast(enabled=True):
                image = self.vae.decode(latents, return_dict=False)[0]  # b 3 512 512 
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload all models
        self.maybe_free_model_hooks()


        if not return_dict:  # t
            if 'ts_module' in cfg.train.model:
                return (image, val_ocr_result)
            else:
                return (image,)

        return StableDiffusion3PipelineOutput(images=image)
