mkdir -p preset
mkdir -p preset/models/llava_ckpt

# download CLIP ViT-L/14-336
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir preset/models/llava_ckpt/clip-vit-large-patch14-336

# download LLaVA-1.5 13B
huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir preset/models/llava_ckpt/llava-v1.5-13b
