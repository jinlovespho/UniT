import glob
import os
import cv2

dataset = 'realtext'

def load_images(path_pattern):
    files = sorted(glob.glob(path_pattern))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}

# Define experiments in a dictionary
experiments = {
    'stage3_straight': f'result_train/stage3/fp16_stage3_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24/realtext/final_result/*.jpg',
    'stage3_stagewise': f'result_train/stage3/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24/realtext/final_result/*.jpg'
}

# Load all experiments
loaded = {name: load_images(path) for name, path in experiments.items()}

# Get common IDs across all experiments
common_ids = sorted(set.intersection(*(set(d.keys()) for d in loaded.values())))

# Save directory
save_dir = f'./vis/realtext/stage3_stagewise_comparison'
os.makedirs(save_dir, exist_ok=True)

# Visualization
for img_id in common_ids:
    imgs = [cv2.imread(exp[img_id]) for exp in loaded.values()]
    img = cv2.vconcat(imgs)  # vertical concat
    cv2.imwrite(f'{save_dir}/{img_id}.jpg', img)

print("FINISH!")




# import glob
# import os
# import cv2

# dataset = 'realtext'

# def load_images(path_pattern):
#     files = sorted(glob.glob(path_pattern))
#     return {os.path.splitext(os.path.basename(f))[0]: f for f in files}

# # Define experiments in a dictionary
# experiments = {
#     'stage2_ocr1': f'result_train/stage2/fp16_stage2_testr_1e-04__ocrloss1.0_descriptive_DiTfeat24_dit4sr_lr1e5_ckpt18k_testr_pretrained/{dataset}/final_result/*.jpg',
#     'stage2_ocr001': f'result_train/stage2/fp16_stage2_testr_1e-04__ocrloss0.01_descriptive_DiTfeat24_dit4sr_lr1e5_ckpt18k_testr_pretrained/{dataset}/final_result/*.jpg'
# }

# # Load all experiments
# loaded = {name: load_images(path) for name, path in experiments.items()}

# # Get common IDs across all experiments
# common_ids = sorted(set.intersection(*(set(d.keys()) for d in loaded.values())))

# # Save directory
# save_dir = f'./vis/{dataset}/stage2_ocr_loss_comparison'
# os.makedirs(save_dir, exist_ok=True)

# # Visualization
# for img_id in common_ids:
#     imgs = [cv2.imread(exp[img_id]) for exp in loaded.values()]
#     img = cv2.vconcat(imgs)  # vertical concat
#     cv2.imwrite(f'{save_dir}/{img_id}.jpg', img)

# print("FINISH!")

