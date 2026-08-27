import os
import numpy as np
from PIL import Image
import odl
from scipy.ndimage import (binary_erosion, binary_dilation,
                           binary_fill_holes, label, generate_binary_structure)


## 640geo CT几何参数定义
class initialization:
    def __init__(self):
        self.param = {}
        self.reso = 512 / 416 * 0.03  # 空间分辨率

        # 图像参数
        self.param['nx_h'] = 416  # 图像横向像素数
        self.param['ny_h'] = 416  # 图像纵向像素数
        self.param['sx'] = self.param['nx_h'] * self.reso  # 图像横向视野
        self.param['sy'] = self.param['ny_h'] * self.reso  # 图像纵向视野

        # 扫描角度参数
        self.param['startangle'] = 0  # 起始角度
        self.param['endangle'] = 2 * np.pi  # 结束角度（全周扫描）
        self.param['nProj'] = 640  # 投影数量

        # 探测器参数
        self.param['su'] = 2 * np.sqrt(self.param['sx'] ** 2 + self.param['sy'] ** 2)  # 探测器长度
        self.param['nu_h'] = 641  # 探测器像素数量
        self.param['dde'] = 1075 * self.reso  # 探测器到物体距离
        self.param['dso'] = 1075 * self.reso  # 源到物体距离

        self.param['u_water'] = 0.192  # 水的线性衰减系数


def build_gemotry(param):
    """构建扇形束CT几何和射线变换算子"""
    # 重建空间（图像域）
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0],
        shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32'
    )

    # 角度分区
    angle_partition = odl.uniform_partition(
        param.param['startangle'],
        param.param['endangle'],
        param.param['nProj']
    )

    # 探测器分区
    detector_partition_h = odl.uniform_partition(
        -(param.param['su'] / 2.0),
        (param.param['su'] / 2.0),
        param.param['nu_h']
    )

    # 扇形束几何
    geometry_h = odl.tomo.FanBeamGeometry(
        angle_partition,
        detector_partition_h,
        src_radius=param.param['dso'],
        det_radius=param.param['dde']
    )

    # 射线变换算子（使用CUDA加速）
    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    return ray_trafo_hh


# 初始化CT系统参数和算子
param = initialization()
ray_trafo = build_gemotry(param)  # 射线变换（正投影）
fbp = odl.tomo.fbp_op(ray_trafo)  # 反投影重建算子（FBP）


def png_to_ct(png_path, hu_min=-1000, hu_max=3000):
    """将PNG图像转换为模拟CT值（HU）"""
    try:
        img = Image.open(png_path).convert('L')  # 转为灰度图
        # 调整尺寸为416x416（与CT几何匹配）
        img = img.resize((param.param['nx_h'], param.param['ny_h']), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        # 映射PNG像素值到HU范围
        ct_img = (img_np / 255.0) * (hu_max - hu_min) + hu_min
        return ct_img
    except Exception as e:
        print(f"处理图像 {png_path} 失败: {e}")
        return None


def detect_closed_regions(ct_img, hu_threshold=-500, min_area=500):
    """
    检测CT图像中的闭合前景区域
    步骤：阈值分割 -> 形态学处理 -> 填充空洞 -> 连通组件分析 -> 筛选有效区域
    """
    # 1. 阈值分割获取初始前景
    foreground = np.where(ct_img > hu_threshold, 1.0, 0.0).astype(np.bool_)

    # 2. 形态学预处理：去除小噪点
    struct = generate_binary_structure(2, 2)  # 8连通结构
    foreground = binary_erosion(foreground, structure=struct, iterations=2)
    foreground = binary_dilation(foreground, structure=struct, iterations=2)

    # 3. 填充空洞，确保区域闭合
    closed_regions = binary_fill_holes(foreground)

    # 4. 连通组件分析
    labeled_regions, num_regions = label(closed_regions, structure=struct)

    # 5. 筛选面积足够大的闭合区域
    valid_regions = np.zeros_like(labeled_regions, dtype=np.float32)
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_area = np.sum(region_mask)

        # 只保留面积足够大的区域（确保是有意义的闭合结构）
        if region_area >= min_area:
            valid_regions[region_mask] = 1.0

    return valid_regions


def generate_metal_in_closed_region(closed_regions_mask, metal_size_range=(10, 50)):
    """
    在闭合区域内生成金属伪影
    确保金属完全位于闭合区域内部，且每个图像至少有一个金属
    """
    shape = closed_regions_mask.shape
    metal_mask = np.zeros(shape, dtype=np.float32)

    # 获取所有闭合区域的坐标
    closed_coords = np.argwhere(closed_regions_mask == 1.0)
    if len(closed_coords) == 0:
        raise ValueError("未检测到有效的闭合前景区域，无法生成金属伪影")

    # 确保至少生成一个金属
    max_attempts = 5  # 最大尝试次数
    attempts = 0
    while np.sum(metal_mask) == 0 and attempts < max_attempts:
        attempts += 1
        # 随机选择闭合区域内的点作为金属中心
        idx = np.random.randint(0, len(closed_coords))
        cx, cy = closed_coords[idx]

        # 随机金属大小
        size = np.random.randint(*metal_size_range)

        # 生成圆形金属区域
        y, x = np.ogrid[-cy:shape[0] - cy, -cx:shape[1] - cx]
        circle = x * x + y * y <= (size // 2) ** 2

        # 确保金属完全在闭合区域内
        valid_metal_area = circle & (closed_regions_mask == 1.0)
        metal_mask[valid_metal_area] = 1.0

    if np.sum(metal_mask) == 0:
        raise RuntimeError(f"尝试{max_attempts}次后仍无法在闭合区域内生成有效金属")

    return metal_mask


def synthesize_metal_artifact(ct_img, metal_mask, metal_hu=3000):
    """合成含金属伪影的CT图像"""
    # 1. 合成含金属的"真实"图像（无伪影的理想情况）
    ct_with_metal = ct_img * (1 - metal_mask) + metal_hu * metal_mask

    # 2. 计算含金属的正弦图（投影），并转换为numpy数组
    sinogram_with_metal = ray_trafo(ct_with_metal).asarray()

    # 3. 模拟金属导致的投影异常（光子饥饿+射线硬化）
    metal_proj = ray_trafo(metal_mask).asarray()
    metal_proj_mask = metal_proj > 0.1  # 金属投影区域掩码

    # 金属区域投影值饱和（模拟光子饥饿）
    proj_max = 4.0
    sinogram_with_metal[metal_proj_mask] = np.clip(
        sinogram_with_metal[metal_proj_mask] * 1.5, 0, proj_max * 0.95
    )

    # 4. 重建带伪影的图像
    ct_with_artifact = fbp(sinogram_with_metal)

    # 转换为numpy数组
    if hasattr(ct_with_artifact, 'asarray'):
        ct_with_artifact = ct_with_artifact.asarray()

    return ct_with_metal, ct_with_artifact, sinogram_with_metal


def normalize_to_png(img, hu_min=-1000, hu_max=3000):
    """将CT值归一化到0~255，用于保存为PNG"""
    img_clamped = np.clip(img, hu_min, hu_max)
    img_norm = (img_clamped - hu_min) / (hu_max - hu_min) * 255
    return img_norm.astype(np.uint8)


def process_folder(input_dir, output_dir, metal_size_range=(10, 50),
                   hu_threshold=-500, min_region_area=500, overwrite=False):
    """
    批量处理文件夹中的PNG图像
    将生成的伪影图像直接保存到指定的输出文件夹
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 遍历输入文件夹中的图像文件
    file_count = 0
    success_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_count += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 检查是否已处理
            if os.path.exists(output_path) and not overwrite:
                print(f"({file_count}) 已跳过 {filename}（文件已存在）")
                continue

            try:
                # 1. 转换为CT图像
                ct_img = png_to_ct(input_path)
                if ct_img is None:
                    continue

                # 2. 检测闭合前景区域
                closed_regions = detect_closed_regions(
                    ct_img,
                    hu_threshold=hu_threshold,
                    min_area=min_region_area
                )

                # 3. 在闭合区域内生成金属掩码（确保至少一个）
                metal_mask = generate_metal_in_closed_region(
                    closed_regions,
                    metal_size_range=metal_size_range
                )

                # 4. 合成伪影
                _, ct_with_artifact, _ = synthesize_metal_artifact(ct_img, metal_mask)

                # 5. 保存伪影图像（直接保存到输出文件夹）
                artifact_png = normalize_to_png(ct_with_artifact)
                Image.fromarray(artifact_png).save(output_path)

                success_count += 1
                print(f"({file_count}) 已处理 {filename}（成功）")

            except Exception as e:
                print(f"({file_count}) 处理 {filename} 失败: {str(e)}")

    print(f"批量处理完成！共处理 {file_count} 个图像文件，成功 {success_count} 个")


if __name__ == "__main__":
    # 配置路径
    input_directory = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images/CT"
    output_directory = "/caoluyang/data/GMAI___SA-Med2D-20M/raw/SAMed2Dv1/merged_classified_images/CT_metal_artifacts"

    # 批量处理
    process_folder(
        input_dir=input_directory,
        output_dir=output_directory,
        metal_size_range=(15, 25),  # 金属尺寸范围
        hu_threshold=-400,  # 前景检测阈值
        min_region_area=800,  # 最小闭合区域面积（像素）
        overwrite=True
    )
