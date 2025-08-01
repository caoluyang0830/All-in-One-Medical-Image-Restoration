import os
import random
import numpy as np
from PIL import Image
from skimage.transform import radon, iradon
from scipy.interpolate import interp1d

# 输入输出文件夹路径
input_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images/CT"
output_folder = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images/CT_metal_artifacts"
os.makedirs(output_folder, exist_ok=True)


# 读取PNG图像（0-255灰度图）
def load_png_images(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')  # 0-255灰度图
            img_array = np.array(img)
            images.append(img_array)
            filenames.append(filename)
    return np.array(images), filenames


# 修正归一化：适配PNG的0-255范围
def normalized(X):
    maxX = 255
    minX = 0
    X = np.clip(X, minX, maxX)
    X = (X - minX) / (maxX - minX)  # 归一化到0~1
    return X


# 生成伪影（修正数值范围）
def Ma(gt):
    rows, cols = gt.shape[:2]
    metalMask = np.zeros_like(gt)
    vec = np.linspace(-cols // 2, cols // 2 - 1, cols)
    xx, yy = np.meshgrid(vec, vec)

    # 金属区域（5-15像素半径）
    num_metals = random.randint(2, 4)
    for _ in range(num_metals):
        x = random.randint(-cols // 2, cols // 2)
        y = random.randint(-rows // 2, rows // 2)
        r = random.randint(5, 15)
        ellipse_ratio = random.uniform(0.7, 1.3)
        metalMask[((xx - x) / ellipse_ratio) ** 2 + (yy - y) ** 2 <= r ** 2] = 1

    # Radon变换（360个角度）
    radon_theta = np.linspace(0, 180, 360)
    R_metal = radon(metalMask, theta=radon_theta, circle=False)
    R_withMetal = radon(gt, theta=radon_theta, circle=False) + R_metal

    # 投影饱和
    max_proj = np.percentile(R_withMetal, 98)
    R_withMetal_saturated = np.clip(R_withMetal, 0, max_proj)

    # 伪影强度参数
    metal_size = np.sum(metalMask)
    a = 10 + 0.01 * metal_size  # 基于0~1范围调整强度
    c = 0.01 + 0.0001 * metal_size

    # 伪影函数（适配0~1范围）
    x = np.linspace(a - 10, a + 10, 100)
    h = (x - (2 * c * (x - a) + 1 - np.sqrt(np.maximum(4 * c * (x - a) + 1, 0)))
         / (2 * c) * np.sign(x - a)) * np.exp(-0.02 * np.abs(x - a))
    h_func = interp1d(x, h, kind='cubic', fill_value="extrapolate")

    # 应用变换
    prj_metalSim = R_withMetal_saturated.copy()
    for i in range(prj_metalSim.shape[0]):
        prj_metalSim[i, :] = h_func(prj_metalSim[i, :])

    # 反Radon变换+范围调整
    metal_image = iradon(prj_metalSim, theta=radon_theta, circle=False)
    # 限制范围并缩放至0~1.5（避免过低）
    metal_image_clamped = np.clip(metal_image, -0.5, 2.0)
    metal_image_scaled = (metal_image_clamped + 0.5) / 2.5 * 1.5  # 映射到0~1.5

    # 噪声
    noise_level = 0.03 * np.max(metal_image_scaled)
    metal_image_noisy = metal_image_scaled + np.random.normal(0, noise_level, size=metal_image_scaled.shape)

    return metal_image_noisy


# 主流程（修正映射到0~255）
def add_metal_artifact_to_folder():
    images, filenames = load_png_images(input_folder)
    if images.size == 0:
        print("文件夹中未找到PNG图像")
        return

    for i in range(images.shape[0]):
        print(f"处理第 {i + 1}/{len(images)} 张图像: {filenames[i]}")
        ori = normalized(images[i, :, :])  # 0-255→0-1
        with_mask = Ma(ori)  # 输出0~1.5范围

        # 映射到0~255（适配PNG）
        with_mask_clamped = np.clip(with_mask, 0, 1.5)  # 截断极端值
        with_mask_scaled = (with_mask_clamped * 255 / 1.5).astype(np.uint8)  # 0~1.5→0~255

        output_path = os.path.join(output_folder, filenames[i])
        Image.fromarray(with_mask_scaled).save(output_path)
        print(f"已保存至: {output_path}")

    print("所有图像处理完成！")


if __name__ == "__main__":
    add_metal_artifact_to_folder()