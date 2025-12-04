import os 
import json
import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



eval_path_list = [
    
    # # pho 5
    # "result_vlm/lq_caption/satext_lv3/satext_lv3_Englishques0/qwenvl_7b",
    
    # # pho6
    # "result_vlm/lq_caption/satext_lv3/satext_lv3_Englishques1/qwenvl_7b",
    
    
    # # pho4
    # "result_vlm/lq_caption/satext_lv3/satext_lv3_Englishques2/qwenvl_7b",
    
    # pho7
    "result_vlm/lq_caption/satext_lv3/satext_lv3_Englishques3/qwenvl_7b"
    
    # # pho7
    # "result_vlm/lq_caption/satext_lv3/satext_lv3_Originalques/qwenvl_7b",
    
]


# load satext_lv3 gt text annotation
ann_path = "/mnt/dataset1/text_restoration/100K/test/dataset.json"
anns = json.load(open(ann_path, 'r'))



# load the tokenizer and the LLM model
# model_name = "Qwen/Qwen3-14B-AWQ
model_name = 'Qwen/Qwen3-14B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)



for eval_path in eval_path_list:
    
    
    # save path 
    lq_extract_save_path = f"{eval_path}/lq_extract_statistic_2nd"
    os.makedirs(lq_extract_save_path, exist_ok=True)
    
    
    # load vlm lq text extraction result
    eval_dict={}
    eval_txt_paths = sorted(glob.glob(f'{eval_path}/*.txt'))
    for eval_txt_path in eval_txt_paths:
        eval_id = eval_txt_path.split('/')[-1].split('.')[0]
        with open(eval_txt_path, 'r') as file:
            lines = file.readlines()
            assert len(lines) == 1
            eval_dict[eval_id] = lines[0]
            

    count_one=0
    count_two=0
    count_three=0
    
    print('Evaluating path: ', eval_path)
    for idx, img_id in enumerate(tqdm(eval_dict.keys())):
        # if idx == 3:
        #     break
        
        # load gt text
        gt_text=[]
        for ann in anns[img_id]['0']['text_instances']:
            gt_text.append(ann['text'])
        gt_text = " ".join(gt_text)

        # load vlm output
        vlm_output = eval_dict[img_id]

        # prepare the model input
        prompt = f"""
        Ground truth text: "{gt_text}"
        VLM OCR output: "{vlm_output}"

        Step 1: Extract the text content from the VLM OCR output.
        Step 2: Compare the extracted text with the ground truth, considering:
        - Word order does NOT matter.
        - Compare based only on the set of unique words in the ground truth.
        - Ignore capitalization, punctuation, and extra/missing spaces.
        - Small typos still count as matches.

        Categories:
        1 — Correct: all unique ground truth words appear in the OCR output (ignoring order, case, spacing, typos).
        2 — Slightly correct (partially correct): at least one but not all unique words match.
        3 — Incorrect: no words match, or the output is largely wrong, unrelated, or empty.

        Answer with only the category number (1, 2, or 3).
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)

        try:
            decision = int(content)
        except ValueError:
            decision = 3  # default to Incorrect if output isn't a number
        if decision == 1:
            count_one += 1 
        elif decision == 2:
            count_two +=1 
        elif decision == 3:
            count_three +=1 

        print(count_one)
        print(count_two)
        print(count_three)

        
        with open(f'{lq_extract_save_path}/{img_id}.txt', 'w') as file:
            file.write(f'img id: {img_id}\n\n')
            file.write(f'gt_text: {gt_text}\n')
            file.write(f'vlm_output: {vlm_output}\n\n')
            file.write(f'----------------------------\n\n')
            file.write(f'{thinking_content}\n')
            file.write(f'LLM classification result: {content}\n')
            file.write(f'1. Correct Num: {count_one}\n')
            file.write(f'2. Slightly Correct Num: {count_two}\n')
            file.write(f'3. Incorrect Num: {count_three}\n')

        # breakpoint()



    total = len(eval_dict.keys())
    p1 = count_one / total * 100
    p2 = count_two / total * 100
    p3 = count_three / total * 100

    final_stat_path = f'{lq_extract_save_path}/final_statistic.txt'
    with open(f'{final_stat_path}', 'w') as file:
        file.write("=== VLM LQ OCR Evaluation Statistics ===\n\n")
        # file.write(f'Qwen2.5 VL ({model_size}b) LQ OCR Result using Qwen3(14b)\n')
        file.write(f"Total images processed: {total}\n\n")

        file.write(f"1. Correct Num: {count_one} ({p1:.2f}%)\n")
        file.write(f"2. Slightly Correct Num: {count_two} ({p2:.2f}%)\n")
        file.write(f"3. Incorrect Num: {count_three} ({p3:.2f}%)\n\n")

        file.write("=== Accuracy Metrics ===\n")
        file.write(f"Exact Accuracy: {p1:.2f}%\n")
        file.write(f"Lenient Accuracy (Correct + Slightly Correct): {(p1+p2):.2f}%\n")
        file.write(f"Incorrect Accuracy: {p3:.2f}%\n")


    # print(f'VLM LQ OCR RESULT for Qwen2.5VL({model_size}b)')
    print(count_one)
    print(count_two)
    print(count_three)


print(f'ALL DONE!')
# breakpoint()
