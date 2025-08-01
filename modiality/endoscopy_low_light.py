import os
import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # 构建Gamma校正查找表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def reduce_illumination(image, reduction_factor=1.0):
    # 将图像转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 降低V通道(亮度)的值
    hsv[:, :, 2] = hsv[:, :, 2] * reduction_factor
    # 转换回BGR色彩空间
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_first_50_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_count = 0

    # 遍历输入文件夹中的所有文件
    for filename in sorted(os.listdir(input_folder)):
        # if processed_count >= 50:
        #     break

        # 检查文件是否为图像
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图像
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)

            if image is not None:
                # 应用较强的Gamma校正使图像变暗 (0.3-0.6范围)
                gamma = np.random.uniform(0.4, 0.7)
                gamma_corrected = adjust_gamma(image, gamma)

                # 应用较强的亮度降低 (0.2-0.4范围)
                reduction = np.random.uniform(0.3, 0.5)
                final_image = reduce_illumination(gamma_corrected, reduction)

                # 保存处理后的图像
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, final_image)

                processed_count += 1
                print(f"已暗化处理 [{processed_count}/50]: {filename} (Gamma: {gamma:.2f}, 亮度降低: {reduction:.2f})")
            else:
                print(f"无法读取图像: {filename}")


# 使用示例
input_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/images"  # 替换为你的输入文件夹路径
output_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/images_dark"  # 替换为你的输出文件夹路径

process_first_50_images(input_folder, output_folder)
