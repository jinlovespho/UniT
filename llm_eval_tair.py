import os 
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-14B-AWQ"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


pred_txts = sorted(glob.glob('results/satext/lv3/tair_tsprompt/txt/*.txt'))
gt_txts = sorted(glob.glob('results/satext/lv3/tair_gtprompt/txt/*.txt'))
assert len(pred_txts) == len(gt_txts), 'check number of pred and gt txts!'


# pred_txts = pred_txts[:3]
# gt_txts = gt_txts[:3]

# all 50 steps
timesteps = [999, 979, 958, 938, 917, 897, 877, 856, 836, 816, 795, 775, 754, 734, 714, 693, 673, 652, 632, 612, 591, 571, 550, 530, 510, 489, 469, 449, 428, 408, 387, 367, 347, 326, 306, 285, 265, 245, 224, 204, 183, 163, 143, 122, 102, 82, 61, 41, 20, 0]

for t in timesteps:

    count_one=0
    count_two=0
    count_three=0

    txts = zip(pred_txts, gt_txts)
    print(f'-- EVALUATING FOR {len(pred_txts)} IMGS -- ')
    for idx, (pred_txt, gt_txt) in enumerate(txts):

        img_id = pred_txt.split('/')[-1].split('.')[0]
        img_id2 = pred_txt.split('/')[-1].split('.')[0]
        assert img_id == img_id2, 'not the same img id'

        # read pred txts
        with open(pred_txt, "r") as pred_file:
            pred_lines = pred_file.readlines()
        pred_dict = {}
        i = 0
        while i < len(pred_lines):
            line = pred_lines[i].strip()
            if line and line.startswith("timestep:"):
                # Extract timestep
                timestep = int(line.split("/")[0].split(":")[1].strip())
                # Extract predicted text list
                pred_text = eval(line.split("/  ts_pred_text:")[1].strip())
                pred_dict[timestep] = pred_text
            i += 1
        # Optional: sort dictionary by timestep descending
        pred_dict = dict(sorted(pred_dict.items(), reverse=True))


        # Read gt txts
        with open(gt_txt, "r") as gt_file:
            gt_lines = gt_file.readlines()
        gt_dict = {}
        i = 0
        while i < len(gt_lines):
            line = gt_lines[i].strip()
            if line and line.startswith("timestep:"):
                # Extract timestep
                timestep = int(line.split("/")[0].split(":")[1].strip())
                # Extract predicted text list
                pred_text = eval(line.split("/  gt_text:")[1].strip())
                gt_dict[timestep] = pred_text
            i += 1
        # Optional: sort dictionary by timestep descending
        gt_dict = dict(sorted(gt_dict.items(), reverse=True))



        gt_text = gt_dict[t]
        vlm_output = pred_dict[t]

        
        

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

        statistic_path = f'results/satext/lv3/vlm_ocr_result/statistic/tair_promptv2/timestep{t}'
        os.makedirs(statistic_path, exist_ok=True)
        with open(f'{statistic_path}/{img_id}.txt', 'w') as file:
            file.write(f'{idx} img id: {img_id} / timestep: {t}\n\n')
            file.write(f'gt_text: {gt_text}\n')
            file.write(f'tair_output: {vlm_output}\n\n')
            file.write(f'----------------------------\n\n')
            file.write(f'{thinking_content}\n')
            file.write(f'LLM classification result: {content}\n')
            file.write(f'1. Correct Num: {count_one}\n')
            file.write(f'2. Slightly Correct Num: {count_two}\n')
            file.write(f'3. Incorrect Num: {count_three}\n')

    total = len(pred_txts)
    p1 = count_one / total * 100
    p2 = count_two / total * 100
    p3 = count_three / total * 100

    final_stat_path = f'{statistic_path}/final_statistic.txt'
    with open(f'{statistic_path}/final_stat_tair_timestep{t}.txt', 'w') as file:
        file.write("=== TAIR TS Module LQ OCR Evaluation Statistics ===\n\n")
        file.write(f"Total images processed: {total}\n\n")

        file.write(f"1. Correct Num: {count_one} ({p1:.2f}%)\n")
        file.write(f"2. Slightly Correct Num: {count_two} ({p2:.2f}%)\n")
        file.write(f"3. Incorrect Num: {count_three} ({p3:.2f}%)\n\n")

        file.write("=== Accuracy Metrics ===\n")
        file.write(f"Exact Accuracy: {p1:.2f}%\n")
        file.write(f"Lenient Accuracy (Correct + Slightly Correct): {(p1+p2):.2f}%\n")
        file.write(f"Incorrect Accuracy: {p3:.2f}%\n")


    print(f'TAIR LQ OCR RESULT for timestep{t}')
    print(count_one)
    print(count_two)
    print(count_three)


print(f'ALL DONE!')
breakpoint()