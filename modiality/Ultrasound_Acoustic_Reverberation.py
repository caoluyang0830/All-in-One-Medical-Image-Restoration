import cv2
import numpy as np
import os
import random


def add_random_ultrasound_artifacts(input_path, output_path):
    """
    使用随机参数添加超声伪影

    参数:
    input_path: 输入图像路径
    output_path: 输出图像路径
    """

    # 读取原始图像
    original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"无法读取图像: {input_path}")

    img = original.copy().astype(np.float32)
    height, width = img.shape

    # ===================== 随机参数生成 =====================
    # 混响参数 (随机范围)
    reverberation_intensity = random.uniform(0.6, 0.9)  # 中等强度混响
    num_echoes = random.randint(2, 5)  # 2-5次回声
    echo_spacing = random.randint(10, 25)  # 间距10-25像素

    # 声影参数 (随机范围)
    shadow_intensity = random.uniform(0.15, 0.4)  # 较暗阴影
    shadow_width = random.randint(60, 120)  # 宽度60-120像素

    # ===================== 混响伪影 =====================
    # 随机反射界面位置 (图像上半部分)
    reflector_y = random.randint(10, height // 3)

    # 随机选择反射界面类型 (水平线或曲线)
    if random.random() > 0.5:
        # 水平线反射
        cv2.line(img, (0, reflector_y), (width, reflector_y), 255, random.randint(2, 4))
    else:
        # 曲线反射 (正弦波)
        x = np.arange(width)
        # 修复：添加了缺失的闭合括号
        y_offset = reflector_y + (np.sin(x / width * 4 * np.pi) * random.randint(5, 15))
        for x_pos in range(width):
            y_pos = min(max(int(y_offset[x_pos]), 0), height - 1)
            img[y_pos, x_pos] = 255

    # 生成混响回声
    for i in range(1, num_echoes + 1):
        echo_y = reflector_y + i * echo_spacing
        if echo_y < height:
            decay_factor = reverberation_intensity ** i
            img[echo_y, :] = np.maximum(img[echo_y, :], img[reflector_y, :] * decay_factor)

            # 随机扩散效果
            for dy in range(-random.randint(1, 3), random.randint(1, 3)):
                if 0 <= echo_y + dy < height:
                    # 修复：添加了缺失的闭合括号
                    img[echo_y + dy, :] = np.maximum(
                        img[echo_y + dy, :],
                        img[reflector_y, :] * decay_factor * random.uniform(0.6, 0.9)
                    )

    # ===================== 声影 =====================
    # 随机声影起始位置
    shadow_x = random.randint(width // 4, 3 * width // 4)
    shadow_y = random.randint(height // 3, 2 * height // 3)

    # 随机反射体形状
    shape_choice = random.choice(["ellipse", "rectangle", "irregular"])

    if shape_choice == "ellipse":
        cv2.ellipse(img, (shadow_x, shadow_y),
                    (random.randint(20, 50), random.randint(10, 30)),
                    0, 0, 360, 255, -1)
    elif shape_choice == "rectangle":
        cv2.rectangle(img,
                      (shadow_x - random.randint(20, 40), shadow_y - random.randint(10, 20)),
                      (shadow_x + random.randint(20, 40), shadow_y + random.randint(10, 20)),
                      255, -1)
    else:  # irregular
        # 修复：调整了点的生成方式，确保数组维度正确
        points = np.array([
            (shadow_x + random.randint(-30, 30), shadow_y + random.randint(-20, 20))
            for _ in range(random.randint(4, 8))
        ], np.int32)
        # 确保点形成一个闭合多边形
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [points], 255)

    # 随机声影形状 (直线或发散)
    shadow_type = random.choice(["uniform", "diverging"])

    # 修复：添加了边界检查，避免除零错误
    max_y_distance = height - shadow_y - 1
    if max_y_distance <= 0:
        max_y_distance = 1  # 避免除零

    for y in range(shadow_y, height):
        if shadow_type == "uniform":
            current_width = shadow_width
        else:  # diverging
            # 修复：添加了缺失的闭合括号
            current_width = int(shadow_width * (1 + random.uniform(0.3, 0.8) * (y - shadow_y) / max_y_distance))

        start_x = max(0, shadow_x - current_width // 2)
        end_x = min(width, shadow_x + current_width // 2)

        # 随机阴影渐变
        depth_factor = (y - shadow_y) / max_y_distance
        attenuation = shadow_intensity + (1 - shadow_intensity) * depth_factor * random.uniform(0.8, 1.2)
        img[y, start_x:end_x] *= max(0, min(attenuation, 1))  # 确保在0-1范围内

    # 保存结果
    result = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, result)
    return os.path.exists(output_path)


# 使用示例
if __name__ == "__main__":
    input_image = "input.png"
    output_image = "output_random_artifacts.png"

    try:
        if add_random_ultrasound_artifacts(input_image, output_image):
            print(f"成功生成带随机伪影的图像: {output_image}")
        else:
            print("图像生成失败")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
