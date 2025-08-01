import os
import numpy as np
import cv2
from tqdm import tqdm
import random


def simulate_lq_pet(image, dose_reduction_factor=12):
    """
    模拟低剂量PET图像退化(剂量减少因子为12)
    参数:
        image: 输入PET图像(0-255范围)
        dose_reduction_factor: 剂量减少因子(默认为12)
    返回:
        退化后的PET图像
    """
    # 将图像转换为浮点型并归一化
    img_float = image.astype(np.float32) / 255.0

    # 添加泊松噪声(模拟低计数率)
    mean_val = np.mean(img_float)
    noisy_img = np.random.poisson(img_float * dose_reduction_factor) / dose_reduction_factor

    # 由于剂量减少，信噪比降低，添加高斯噪声模拟重建误差
    noisy_img += np.random.normal(0, mean_val / 4, size=img_float.shape)

    # 确保值在0-1之间
    noisy_img = np.clip(noisy_img, 0, 1)

    # 由于低剂量图像通常分辨率较低，添加模糊效果
    kernel_size = 3
    noisy_img = cv2.GaussianBlur(noisy_img, (kernel_size, kernel_size), 0)

    # 转换回0-255范围
    noisy_img = (noisy_img * 255).astype(np.uint8)

    return noisy_img


def process_folder(input_folder, output_folder, dose_reduction_factor=12):
    """
    处理文件夹中的所有PET图像
    参数:
        input_folder: 输入图像文件夹路径
        output_folder: 输出图像文件夹路径
        dose_reduction_factor: 剂量减少因子
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有png文件
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    for filename in tqdm(file_list, desc="Processing PET images"):
        # 读取图像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 假设是灰度图像

        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # 模拟低剂量PET
        lq_img = simulate_lq_pet(img, dose_reduction_factor)

        # 保存结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, lq_img)


if __name__ == "__main__":
    input_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images/PET"
    output_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images/PET_denoised"  # 替换为你的输出文件夹路径

    # 设置剂量减少因子为12
    dose_reduction_factor = 12

    process_folder(input_folder, output_folder, dose_reduction_factor)

    print("Processing complete!")