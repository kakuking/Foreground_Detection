import numpy as np
import os
from PIL import Image
from tqdm import tqdm

dataset_path = "./Dataset"
dataset_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

X_data = []
Y_data = [] 
for folder in dataset_folders:
    input_folder = os.path.join(folder, "input")
    ground_truth_folder = os.path.join(folder, "groundtruth")

    ip_images = [f.path for f in os.scandir(input_folder) if f.is_file()]
    
    for ip_img in tqdm(ip_images, desc=f"Processing {os.path.basename(folder)}"):
        img = Image.open(ip_img)

        input_arr = np.array(img) / 255
        X_data.append(input_arr)

        gt_name = os.path.basename(ip_img)
        gt_path = os.path.join(ground_truth_folder, "gt" + gt_name[2:-4] + ".png")

        gt_img = Image.open(gt_path)

        gt_arr = np.array(gt_img) / 255
        Y_data.append(gt_arr)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

Y_data = Y_data.reshape(Y_data.shape[0], Y_data.shape[1], Y_data.shape[2], 1)


np.save("X_data.npy", X_data)
np.save("Y_data.npy", Y_data)