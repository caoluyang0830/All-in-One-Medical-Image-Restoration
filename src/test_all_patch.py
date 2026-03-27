import os
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from net.AMIFound import AMIFound_Multiexpers
from net.AMIFound_2 import NestedAMIFound
from options2 import train_options
from utils.test_utils import save_img
from data.dataset_utils_all import IRBenchmarks, CDD11


####################################################################################################
## HELPERS
def check_tensor_size(tensor, name):
    """检查张量是否超过32位索引限制"""
    max_int32 = 2147483647  # 2^31 - 1

    # 检查元素总数
    num_elements = tensor.numel()
    if num_elements > max_int32:
        print(f"⚠️ 警告: 张量 '{name}' 过大 - 元素数量: {num_elements} > {max_int32}")
        return False

    # 检查步长
    for i, stride in enumerate(tensor.stride()):
        if stride > max_int32:
            print(f"⚠️ 警告: 张量 '{name}' 维度 {i} 步长过大: {stride} > {max_int32}")
            return False

    return True
def compute_psnr(image_true, image_test, image_mask, data_range=None):
    # this function is based on skimage.metrics.peak_signal_noise_ratio
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range=1.0,
                                               full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim


def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))


def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range=1.0, full=False)


####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        # self.net = AMIFound(
        #     dim=opt.dim,
        #     num_blocks=opt.num_blocks,
        #     num_dec_blocks=opt.num_dec_blocks,
        #     levels=len(opt.num_blocks),
        #     heads=opt.heads,
        #     num_refinement_blocks=opt.num_refinement_blocks,
        #     # 替换原有单层专家参数为双层结构参数
        #     # num_principals=opt.num_principals,  # 新增：校长数量（对应原专家层级）
        #     # top_k_principals=opt.top_k_principals,  # 新增：每个学生选几个校长
        #     # num_teachers_per_principal=opt.num_experts,  # 映射：原num_experts变为每个校长手下的老师数量
        #     top_k_teachers=opt.topk,  # 映射：原topk变为每个校长下选几个老师
        #     teacher_rank=opt.latent_dim,  # 映射：原rank变为老师的中间维度
        #     teacher_depth=opt.stage_depth,  # 映射：原stage_depth变为老师的处理深度
        #     with_complexity=opt.with_complexity,
        #     complexity_scale=opt.complexity_scale,
        # )
        self.net = AMIFound_Multiexpers(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale, )
        # self.net = NestedAMIFound(
        #     dim=opt.dim,
        #     num_blocks=opt.num_blocks,
        #     num_dec_blocks=opt.num_dec_blocks,
        #     levels=len(opt.num_blocks),
        #     heads=opt.heads,
        #     num_refinement_blocks=opt.num_refinement_blocks,
        #     topk=opt.topk,
        #     num_experts=opt.num_exp_blocks,
        #     rank=opt.latent_dim,
        #     with_complexity=opt.with_complexity,
        #     depth_type=opt.depth_type,
        #     stage_depth=opt.stage_depth,
        #     rank_type=opt.rank_type,
        #     complexity_scale=opt.complexity_scale, )

    def forward(self, x):
        return self.net(x)


# 添加处理大张量的辅助函数
def process_large_tensor(net, tensor, chunk_size=1):
    """分块处理超出32位限制的大张量"""
    try:
        # 尝试按批次维度分块
        if tensor.dim() == 4 and tensor.size(0) > 1:  # [B, C, H, W]
            outputs = []
            for i in range(0, tensor.size(0), chunk_size):
                chunk = tensor[i:i + chunk_size]
                outputs.append(net(chunk))
            return torch.cat(outputs, dim=0)

        # 尝试按空间维度分块
        print("尝试按空间维度分块...")
        return spatial_chunk_processing(net, tensor)

    except RuntimeError as e:
        print(f"分块处理失败: {str(e)}")
        return None


def spatial_chunk_processing(net, tensor, tile_size=512):
    """按空间维度分块处理图像张量 (当h或w > 1800时自动调用)"""
    if tensor.dim() != 4:
        print("空间分块仅支持4D张量 [B,C,H,W]")
        return net(tensor)

    B, C, H, W = tensor.shape

    # 计算最佳分块大小（基于原始tile_size，但不超过图像尺寸）
    actual_tile_size = min(tile_size, H, W)

    # 计算分块数量
    num_tiles_h = (H + actual_tile_size - 1) // actual_tile_size
    num_tiles_w = (W + actual_tile_size - 1) // actual_tile_size

    # 重叠分块以避免边界伪影
    overlap = min(32, actual_tile_size // 4)  # 重叠区域不超过tile_size的1/4

    # 初始化输出张量
    output = torch.zeros_like(tensor)

    for b in range(B):  # 处理每个批次
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # 计算分块位置 (考虑边界)
                h_start = max(0, i * actual_tile_size - overlap)
                h_end = min(H, (i + 1) * actual_tile_size + overlap)
                w_start = max(0, j * actual_tile_size - overlap)
                w_end = min(W, (j + 1) * actual_tile_size + overlap)

                # 提取分块
                tile = tensor[b:b + 1, :, h_start:h_end, w_start:w_end]

                # 处理分块
                processed_tile = net(tile)
                if isinstance(processed_tile, (list, tuple)):
                    processed_tile = processed_tile[0]

                # 计算有效区域 (去除重叠部分)
                crop_top = overlap if i > 0 else 0
                crop_bottom = processed_tile.size(2) - (overlap if i < num_tiles_h - 1 else 0)
                crop_left = overlap if j > 0 else 0
                crop_right = processed_tile.size(3) - (overlap if j < num_tiles_w - 1 else 0)

                # 确保有效区域为正
                if crop_bottom <= crop_top or crop_right <= crop_left:
                    continue

                valid_region = processed_tile[:, :, crop_top:crop_bottom, crop_left:crop_right]

                # 计算输出位置
                out_h_start = i * actual_tile_size
                out_h_end = min(H, (i + 1) * actual_tile_size)
                out_w_start = j * actual_tile_size
                out_w_end = min(W, (j + 1) * actual_tile_size)

                # 确保区域匹配
                valid_h = out_h_end - out_h_start
                valid_w = out_w_end - out_w_start

                if valid_region.size(2) != valid_h or valid_region.size(3) != valid_w:
                    # 如果不匹配，调整有效区域大小
                    valid_region = torch.nn.functional.interpolate(
                        valid_region, size=(valid_h, valid_w), mode='bilinear', align_corners=False
                    )

                # 将有效区域复制到输出
                output[b:b + 1, :, out_h_start:out_h_end, out_w_start:out_w_end] = valid_region

    return output
####################################################################################################
def run_test(opts, net, dataset, factor=8):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=16)

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}")).mkdir(
            parents=True, exist_ok=True)
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    psnr, ssim, lpips = [], [], []
    with torch.no_grad():
        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            # ===== 新增: 检查高度或宽度是否超过1800 =====
            _, _, h, w = degrad_patch.shape
            if h > 2000 or w > 2000:
                print(f"⚠️ 图像尺寸过大 ({h}x{w})，启动分块处理")
                restored = spatial_chunk_processing(net, degrad_patch)
            else:
                restored = net(degrad_patch)
            # ===== 结束新增部分 =====

            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."

            # save output images
            restored = torch.clamp(restored, 0, 1)
            lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())

            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            degrad_patch = degrad_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            ssim.append(calc_ssim(clean, restored))
            psnr_temp = peak_signal_noise_ratio(clean, restored, data_range=1)
            psnr.append(psnr_temp)

            if opts.save_results:
                save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(
                    round(psnr_temp, 2)) + '.png'
                save_img(
                    (os.path.join(os.getcwd(),
                                  f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}",
                                  save_name)),
                    img_as_ubyte(restored))

    print('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr), np.mean(ssim), np.mean(lpips)))


## test LolV1
def run_synllie(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test GoPro
def run_gopro(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test Derain
def run_derain(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test Dehaze
def run_dehaze(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test synthetic denoising
def run_denoise_15(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


def run_denoise_25(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


def run_denoise_50(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

def run_denoise(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

def run_deblur(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)
# test CDD11
def run_cdd11(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

def run_CT(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

def run_MR(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

####################################################################################################
## main
def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load model
    net = PLTestModel.load_from_checkpoint(
        os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt"), opt=opt).cuda()
    net.eval()
    for de in opt.benchmarks:
        ind_opt = opt
        ind_opt.benchmarks = [de]

        if "CDD11" in opt.trainset:
            _, subset = opt.trainset.split("_", maxsplit=1)
            dataset = CDD11(opt, split="test", subset=subset)
        else:
            dataset = IRBenchmarks(ind_opt)

        print("--------> Testing on", de, "testset.")
        print("\n")
        globals()[f"run_{de}"](opt, net, dataset, factor=8)


def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    train_opt = train_options()

    main(train_opt)
