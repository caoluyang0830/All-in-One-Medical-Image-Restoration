import os
import numpy as np
import cv2
from tqdm import tqdm


def kspace_downsample(image, factor=4):
    """
    通过裁剪k空间中心区域生成低分辨率图像
    factor: 下采样因子 (保留中心 1/factor 的数据)
    """
    # 转换为浮点型进行傅里叶变换
    image = image.astype(np.float32) / 255.0

    # 傅里叶变换并中心化
    kspace = np.fft.fft2(image)
    kspace_shifted = np.fft.fftshift(kspace)

    # 获取图像尺寸和中心位置
    h, w = image.shape
    center_h, center_w = h // 2, w // 2

    # 计算裁剪区域大小 (保留中心的 1/factor)
    crop_h = int(h / factor)
    crop_w = int(w / factor)

    # 创建掩码 (保留中心区域)
    mask = np.zeros_like(image, dtype=np.float32)
    mask[
    center_h - crop_h // 2: center_h + crop_h // 2,
    center_w - crop_w // 2: center_w + crop_w // 2
    ] = 1

    # 应用k空间裁剪
    kspace_cropped = kspace_shifted * mask

    # 逆傅里叶变换
    kspace_ishifted = np.fft.ifftshift(kspace_cropped)
    image_recon = np.fft.ifft2(kspace_ishifted)
    image_recon = np.abs(image_recon)

    # 归一化并转换为8位图像
    image_recon = (image_recon - image_recon.min())
    image_recon = image_recon / image_recon.max() * 255
    return image_recon.astype(np.uint8)


# 路径设置
base_path = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images"
# base_path = "D:/Desktop/aircraft"
input_folder = os.path.join(base_path, "MR")
output_folder = os.path.join(base_path, "MR_LQ")

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

# 处理所有PNG图像
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)

        # 读取图像并转为灰度
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"跳过无法读取的图像: {filename}")
            continue

        # 生成退化图像
        degraded_img = kspace_downsample(img, factor=4)

        # 保存结果
        cv2.imwrite(out_path, degraded_img)

print(f"处理完成! 结果保存在: {output_folder}")