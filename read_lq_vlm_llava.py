import os 
import json
import glob
import torch
from PIL import Image 
from transformers import AutoProcessor, LlavaForConditionalGeneration



dataset_list = [
    'realtext', 
    'satext_lv1', 
    'satext_lv2', 
    'satext_lv3',
]

lq_path_list = [
    '/mnt/dataset1/text_restoration/tair_published/real_text/LQ',
    '/mnt/dataset1/text_restoration/SAMText_test_degradation/lv1',
    '/mnt/dataset1/text_restoration/SAMText_test_degradation/lv2',
    '/mnt/dataset1/text_restoration/SAMText_test_degradation/lv3',
    
]

# question_list = [
#     'OCR this image.',
#     "Read and transcribe any text you can find in this low-resolution image.",
#     "Describe what you see in this blurry image, paying special attention to any visible text or characters.",
#     "Extract all readable words and letters from this low-quality image, even if they are unclear.",
# ]


# english focused input prompt
question_list = [
    "OCR this image and transcribe only the English text.",
    "Read and transcribe all English text visible in this low-resolution image.",
    "Describe the contents of this blurry image, focusing only on any visible English text or characters.",
    "Extract all visible English words and letters from this low-quality image, even if they appear unclear.",
]


for dataset, lq_path in zip(dataset_list, lq_path_list):
    
    print(dataset)
    print(lq_path)

    lq_imgs = sorted(glob.glob(f'{lq_path}/*.jpg'))
    
    # load vlm
    # model_size_list=[7, 13]
    model_size_list=[13]
    for model_size in model_size_list:
        
        print(f'llava_{model_size}b')
        model = LlavaForConditionalGeneration.from_pretrained(f"llava-hf/llava-1.5-{model_size}b-hf", torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(f"llava-hf/llava-1.5-{model_size}b-hf", use_fast=True)

        for q_idx, question in enumerate(question_list):
                
            for lq in lq_imgs:
                
                lq_id = lq.split('/')[-1].split('.')[0]
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": f"{lq}"},
                            {"type": "text", "text": question},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device, torch.float16)


                lq_img = Image.open(lq).convert("RGB")
                # Add the actual image tensor
                inputs["pixel_values"] = processor.image_processor([lq_img], return_tensors="pt").pixel_values.to(model.device, torch.float16)


                # Generate
                generate_ids = model.generate(**inputs, max_new_tokens=200)
                output = processor.batch_decode(generate_ids, skip_special_tokens=True)
                # print(output[0])
                # print('-'*50)
                output_result = output[0].split('ASSISTANT')[-1][2:]
                print(output_result)
                
                # save as txt file
                # save_path = f"result_vlm/lq_caption/{dataset}_ques{q_idx}/llava_{model_size}b"
                save_path = f"result_vlm/lq_caption/{dataset}_Englishques{q_idx}/llava_{model_size}b"
                os.makedirs(save_path, exist_ok=True)
                txt_save_path = f"{save_path}/{lq_id}.txt"
                with open(txt_save_path, 'w') as file:
                    file.write(output_result)


print('FINISH!!')