import os
import json
import glob
import torch
from PIL import Image 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


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
    model_size_list=[3, 7, 32, 72]
    for model_size in model_size_list:
        
        print(f'qwenvl_{model_size}b')
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct", torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{model_size}B-Instruct")
        
        for q_idx, question in enumerate(question_list):
            
            for lq in lq_imgs:
                
                lq_id = lq.split('/')[-1].split('.')[0]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"{lq}",
                            },
                            {"type": "text", "text": f"{question}"},
                        ],
                    }
                ]
                
                
                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # print(output_text[0])
                # print('-'*50)
                clean_text = output_text[0].replace('\n', "")
                print(clean_text)
                
                # save as txt file
                # save_path = f"result_vlm/lq_caption/{dataset}_ques{q_idx}/qwenvl_{model_size}b"
                save_path = f"result_vlm/lq_caption/{dataset}_Englishques{q_idx}/qwenvl_{model_size}b"
                os.makedirs(save_path, exist_ok=True)
                txt_save_path = f"{save_path}/{lq_id}.txt"
                with open(txt_save_path, 'w') as file:
                    file.write(clean_text)


print('FINISH!!')