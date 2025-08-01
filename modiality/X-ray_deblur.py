import os
import cv2
import numpy as np
import random


def apply_gaussian_blur(image, kernel_size=None, sigma=None):
    """应用高斯模糊"""
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    if sigma is None:
        sigma = random.uniform(0.5, 2.0)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred, kernel_size, sigma


def apply_mean_blur(image, kernel_size=None):
    """应用均值模糊"""
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    blurred = cv2.blur(image, (kernel_size, kernel_size))
    return blurred, kernel_size


def apply_median_blur(image, kernel_size=None):
    """应用中值模糊"""
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    blurred = cv2.medianBlur(image, kernel_size)
    return blurred, kernel_size


def apply_srad(image, iterations=None, delta=None, kappa=None):
    """
    应用散斑减少各向异性扩散(SRAD)
    基于Perona-Malik各向异性扩散的改进版本，适用于超声图像
    """
    if iterations is None:
        iterations = random.randint(3, 10)
    if delta is None:
        delta = random.uniform(0.01, 0.05)
    if kappa is None:
        kappa = random.uniform(5, 20)

    # 转换为浮点型
    img_float = image.astype(np.float32) / 255.0

    # 初始化扩散图像
    diffused = img_float.copy()

    for _ in range(iterations):
        # 计算梯度
        grad_x = cv2.Sobel(diffused, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(diffused, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # SRAD扩散系数
        q0_squared = (grad_mag ** 2 - (kappa ** 2)) / (kappa ** 4)
        c = 1.0 / (1.0 + q0_squared)

        # 应用扩散
        diffused_x = cv2.Sobel(diffused * c, cv2.CV_32F, 1, 0, ksize=3)
        diffused_y = cv2.Sobel(diffused * c, cv2.CV_32F, 0, 1, ksize=3)
        diffused += delta * (diffused_x + diffused_y)

    # 归一化回0-255范围
    diffused = np.clip(diffused * 255, 0, 255).astype(np.uint8)
    return diffused, iterations, delta, kappa


def apply_all_blurs(image):
    """应用所有四种模糊效果"""
    # 应用高斯模糊
    blurred, g_kernel, g_sigma = apply_gaussian_blur(image)

    # 应用均值模糊
    blurred, m_kernel = apply_mean_blur(blurred)

    # 应用中值模糊
    blurred, med_kernel = apply_median_blur(blurred)

    # 应用SRAD
    final_blurred, srad_iter, srad_delta, srad_kappa = apply_srad(blurred)

    # 收集所有参数
    blur_params = {
        'gaussian': {'kernel': g_kernel, 'sigma': g_sigma},
        'mean': {'kernel': m_kernel},
        'median': {'kernel': med_kernel},
        'srad': {
            'iterations': srad_iter,
            'delta': srad_delta,
            'kappa': srad_kappa
        }
    }

    return final_blurred, blur_params


def process_blur_images(input_folder, output_folder, num_images=None):
    """处理图像，叠加四种模糊效果"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 如果指定了处理数量，则截取相应数量的图像
    if num_images is not None:
        image_files = image_files[:num_images]

    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"无法读取图像: {filename}")
            continue

        # 应用所有模糊效果
        processed, blur_params = apply_all_blurs(image)

        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed)

        # 打印处理信息
        blur_info = (
            f"高斯模糊(k={blur_params['gaussian']['kernel']},σ={blur_params['gaussian']['sigma']:.2f}), "
            f"均值模糊(k={blur_params['mean']['kernel']}), "
            f"中值模糊(k={blur_params['median']['kernel']}), "
            f"SRAD(i={blur_params['srad']['iterations']},δ={blur_params['srad']['delta']:.3f},κ={blur_params['srad']['kappa']:.1f})"
        )

        print(f"[{idx}/{len(image_files)}] 已处理: {filename} ({blur_info})")


# 使用示例
if __name__ == "__main__":
    input_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/images"
    output_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/images_blur"

    # 处理图像并叠加四种模糊效果
    print("正在处理图像并叠加四种模糊效果...")
    process_blur_images(input_folder, output_folder)