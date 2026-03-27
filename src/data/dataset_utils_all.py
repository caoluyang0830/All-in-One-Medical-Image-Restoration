import os
import cv2
import glob
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode

from data.degradation_utils import Degradation
from utils.image_utils import random_augmentation, crop_img


class CDD11(Dataset):
    def __init__(self, args, split: str = "train", subset: str = "all"):
        super(CDD11, self).__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.dataset_split = split
        self.subset = subset
        if split == "train":
            self.patch_size = args.patch_size
        else:
            self.patch_size = 64

        self._init()

    def __getitem__(self, index):
        # Randomly select a degradation type
        if self.dataset_split == "train":
            degradation_type = random.choice(list(self.degraded_dict.keys()))
            degraded_image_path = random.choice(self.degraded_dict[degradation_type])
        else:
            degradation_type = self.subset
            degraded_image_path = self.degraded_dict[degradation_type][index]

        # Select a degraded image within that type

        degraded_name = os.path.basename(degraded_image_path)

        # Get the corresponding clean image based on the file name
        image_name = os.path.basename(degraded_image_path)
        assert degraded_name == image_name
        clean_image_path = os.path.join(os.path.dirname(self.clean[0]), image_name)

        # Load the images
        # lr = crop_img(np.array(Image.open(degraded_image_path).convert('RGB')), base=16)
        lr = np.array(Image.open(degraded_image_path).convert('RGB'))
        # hr = crop_img(np.array(Image.open(clean_image_path).convert('RGB')), base=16)
        hr = np.array(Image.open(clean_image_path).convert('RGB'))
        # Apply random augmentation and crop
        if self.dataset_split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        # Convert to tensors
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [clean_image_path, degradation_type], lr, hr

    def __len__(self):
        return sum(len(images) for images in self.degraded_dict.values())

    def _init(self):
        data_dir = os.path.join(self.args.data_file_dir, "cdd11")
        self.clean = sorted(glob.glob(os.path.join(data_dir, f"{self.dataset_split}/clear", "*.png")))

        if len(self.clean) == 0:
            raise ValueError(f"No clean images found in {os.path.join(data_dir, f'{self.dataset_split}/clear')}")

        self.degraded_dict = {}
        allowed_degradation_folders = self._filter_degradation_folders(data_dir)
        for folder in allowed_degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            degraded_images = sorted(glob.glob(os.path.join(folder, "*.png")))

            if len(degraded_images) == 0:
                raise ValueError(f"No images found in {folder_name}")

            # scale dataset length
            if self.dataset_split == "train":
                degraded_images *= 2

            self.degraded_dict[folder_name] = degraded_images

    def _filter_degradation_folders(self, data_dir):
        """
        This function returns folders based on the degradation_type_mode.
        'single', 'double', 'triple', or 'all' degradation types will be returned.
        """
        degradation_folders = sorted(glob.glob(os.path.join(data_dir, self.dataset_split, "*/")))
        filtered_folders = []

        for folder in degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            if folder_name == "clear":
                continue

            # Count the number of degradations based on the number of underscores in the folder name
            degradation_count = folder_name.count('_') + 1

            # Check the degradation type mode and filter accordingly
            if self.subset == "single" and degradation_count == 1:
                filtered_folders.append(folder)
            elif self.subset == "double" and degradation_count == 2:
                filtered_folders.append(folder)
            elif self.subset == "triple" and degradation_count == 3:
                filtered_folders.append(folder)
            elif self.subset == "all":
                filtered_folders.append(folder)
            # If self.subset is a specific degradation folder name, match it exactly
            elif self.subset not in ["single", "double", "triple", "all"]:
                if folder_name == self.subset:
                    filtered_folders.append(folder)

        print(f"Degradation type mode: {self.subset}")
        print(f"Loading degradation folders: {[os.path.basename(f.strip('/')) for f in filtered_folders]}")
        return filtered_folders

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        # print(H, W)

        # 如果图像太小，先进行填充或缩放
        if H < self.args.patch_size or W < self.args.patch_size:
            # # 方法 1：填充（padding）到至少 patch_size
            # pad_h = max(0, self.args.patch_size - H)
            # pad_w = max(0, self.args.patch_size - W)
            # img_1 = np.pad(img_1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            # img_2 = np.pad(img_2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            # H, W = img_1.shape[:2]  # 更新尺寸

            # 方法 2：缩放（resize）到至少 patch_size
            scale = max(self.args.patch_size / H, self.args.patch_size / W)
            new_H, new_W = int(H * scale), int(W * scale)
            img_1 = cv2.resize(img_1, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            H, W = new_H, new_W

        # 随机选择裁剪位置
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        # 裁剪 patch
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2


class AIOTrainDataset(Dataset):
    """
    Dataset class for training on degraded images (支持CT/MRI按Epoch动态跨步采样).
    """

    def __init__(self, args):
        super(AIOTrainDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}

        # 图像预处理 Pipeline
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),  # 修正：原代码少了transforms.前缀
        ])
        self.toTensor = ToTensor()

        # 新增：Epoch管理（关键参数，需在args中设置total_epochs=200）
        self.total_epochs = args.epochs  # 总训练Epoch数（用户需求：200）
        self.current_epoch = 0  # 当前Epoch，初始为0

        # 初始化所有数据集（CT/MRI会保存完整训练对，不直接生成样本）
        self._init_lr()
        # 初始化时不合并样本，等待第一个Epoch调用set_epoch后合并
        self.current_lr = []
        self.current_hr = []

    def __getitem__(self, idx):
        """获取单个样本（从当前Epoch的动态样本列表中读取）"""
        # 关键修改：从current_lr/current_hr获取当前Epoch样本
        lr_sample = self.current_lr[idx]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]

        if deg_type in ["denoise_15", "denoise_25", "denoise_50"]:
            # 降噪任务：从干净图生成带噪图（原逻辑不变）
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = self.crop_transform(hr)
            hr = np.array(hr)
            hr = random_augmentation(hr)[0]
            lr = self.D.single_degrade(hr, de_id)
        else:
            if deg_type == "dehaze_original":
                # 去雾任务：单独处理干净图路径（原逻辑不变）
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            else:
                # 其他任务（含CT/MRI）：从当前Epoch的hr样本读取
                hr_sample = self.current_hr[idx]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)

            # 裁剪与数据增强（原逻辑不变）
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        # 转为Tensor格式
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [lr_sample["img"], de_id], lr, hr

    def __len__(self):
        """当前Epoch的总样本数（动态变化）"""
        return len(self.current_lr)

    def set_epoch(self, current_epoch):
        """
        关键方法：更新当前Epoch，动态生成CT/MRI样本子集
        参数：current_epoch - 当前训练的Epoch（从0开始）
        """
        self.current_epoch = current_epoch

        # 1. 动态生成当前Epoch的CT样本
        self._generate_current_ct_samples()
        # 2. 动态生成当前Epoch的MRI样本
        self._generate_current_mr_samples()
        # 3. 合并静态数据集与当前Epoch的动态样本
        self._merge_current_tasks()

    def _init_lr(self):
        """初始化所有数据集（CT/MRI保存完整训练对，其他数据集生成静态样本）"""
        # synthetic datasets
        if 'Endoscopy' in self.de_type:
            self._init_synllie(id=self.de_dict['Endoscopy'])
        if 'Fundus' in self.de_type:
            self._init_deblur(id=self.de_dict['Fundus'])
        if 'PET' in self.de_type:
            self._init_derain(id=self.de_dict['PET'])
        if 'Ultrasound' in self.de_type:
            self._init_dehaze(id=self.de_dict['Ultrasound'])
        if 'X-ray' in self.de_type:
            self._init_denoise(id=self.de_dict['X-ray'])
        if 'CT' in self.de_type:
            self._init_CT(id=self.de_dict['CT'])  # 关键：CT保存完整训练对
        if 'MR' in self.de_type:
            self._init_MR(id=self.de_dict['MR'])  # 关键：MRI保存完整训练对
        if 'denoise_15' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_25' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_50' in self.de_type:
            self._init_clean(id=0)

    def _merge_current_tasks(self):
        """合并静态数据集（如Endoscopy）与当前Epoch的CT/MRI动态样本"""
        self.current_lr = []
        self.current_hr = []

        # 1. 添加静态数据集（原逻辑不变，仅修改属性名）
        if hasattr(self, 'synllie_lr') and hasattr(self, 'synllie_hr'):
            self.current_lr += self.synllie_lr
            self.current_hr += self.synllie_hr
        if hasattr(self, 'denoise_lr') and hasattr(self, 'denoise_hr'):
            self.current_lr += self.denoise_lr
            self.current_hr += self.denoise_hr
        if hasattr(self, 's15_ids'):
            self.current_lr += self.s15_ids
            self.current_hr += self.s15_ids
        if hasattr(self, 's25_ids'):
            self.current_lr += self.s25_ids
            self.current_hr += self.s25_ids
        if hasattr(self, 's50_ids'):
            self.current_lr += self.s50_ids
            self.current_hr += self.s50_ids
        if hasattr(self, 'deblur_lr') and hasattr(self, 'deblur_hr'):
            self.current_lr += self.deblur_lr
            self.current_hr += self.deblur_hr
        if hasattr(self, 'derain_lr') and hasattr(self, 'derain_hr'):
            self.current_lr += self.derain_lr
            self.current_hr += self.derain_hr
        if hasattr(self, 'dehaze_lr') and hasattr(self, 'dehaze_hr'):
            self.current_lr += self.dehaze_lr
            self.current_hr += self.dehaze_hr

        # 2. 添加当前Epoch的CT动态样本（_generate_current_ct_samples生成）
        if hasattr(self, 'current_CT_lr') and hasattr(self, 'current_CT_hr'):
            self.current_lr += self.current_CT_lr
            self.current_hr += self.current_CT_hr

        # 3. 添加当前Epoch的MRI动态样本（_generate_current_mr_samples生成）
        if hasattr(self, 'current_MR_lr') and hasattr(self, 'current_MR_hr'):
            self.current_lr += self.current_MR_lr
            self.current_hr += self.current_MR_hr

        # 打印当前Epoch样本统计
        print(f"Epoch {self.current_epoch} - Total samples: {len(self.current_lr)} "
              f"(CT: {len(self.current_CT_lr) if hasattr(self, 'current_CT_lr') else 0}, "
              f"MR: {len(self.current_MR_lr) if hasattr(self, 'current_MR_lr') else 0})")

    def _generate_current_ct_samples(self):
        """生成当前Epoch的CT样本子集（跨步采样）"""
        if not hasattr(self, 'ct_train_pairs'):
            self.current_CT_lr = []
            self.current_CT_hr = []
            return

        # 计算当前Epoch的采样区间
        total = self.total_ct_train  # 总CT训练样本数（7:3分割后）
        batch_size = self.batch_ct  # 每个Epoch的CT样本数
        start_idx = self.current_epoch * batch_size

        # 处理最后一个Epoch：覆盖剩余所有样本（避免遗漏）
        if self.current_epoch == self.total_epochs - 1:
            end_idx = total
        else:
            end_idx = start_idx + batch_size

        # 截取当前Epoch的CT样本对
        current_pairs = self.ct_train_pairs[start_idx:end_idx]
        # 生成lr/hr样本列表（格式与其他数据集一致）
        self.current_CT_lr = [{"img": pair[0], "de_type": self.ct_de_type_id} for pair in current_pairs]
        self.current_CT_hr = [{"img": pair[1], "de_type": self.ct_de_type_id} for pair in current_pairs]

    def _generate_current_mr_samples(self):
        """生成当前Epoch的MRI样本子集（与CT逻辑一致）"""
        if not hasattr(self, 'mr_train_pairs'):
            self.current_MR_lr = []
            self.current_MR_hr = []
            return

        total = self.total_mr_train
        batch_size = self.batch_mr
        start_idx = self.current_epoch * batch_size

        if self.current_epoch == self.total_epochs - 1:
            end_idx = total
        else:
            end_idx = start_idx + batch_size

        current_pairs = self.mr_train_pairs[start_idx:end_idx]
        self.current_MR_lr = [{"img": pair[0], "de_type": self.mr_de_type_id} for pair in current_pairs]
        self.current_MR_hr = [{"img": pair[1], "de_type": self.mr_de_type_id} for pair in current_pairs]

    # ------------------------------
    # 数据集初始化方法（仅CT/MRI有修改）
    # ------------------------------
    def _init_CT(self, id):
        """修改：保存完整CT训练对，计算每个Epoch的采样量"""
        random.seed(42)  # 固定种子确保可复现
        inputs = "/data1/luyang/data/extracted_top50_samples/CT_metal_artifacts"
        targets = "/data1/luyang/data/extracted_top50_samples/CT"

        # 获取文件列表并配对
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))
        if len(lr_files) != len(hr_files):
            print(f"警告: CT文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 7:3分割训练集（仅保存训练对，不生成样本）
        split_index = int(len(paired_files) * 0.7)
        self.ct_train_pairs = paired_files[:split_index]  # 完整训练对列表
        self.ct_de_type_id = id  # 数据集类型ID
        self.total_ct_train = len(self.ct_train_pairs)  # 总CT训练样本数

        # 计算每个Epoch的CT采样量（关键：确保200Epoch覆盖全部）
        self.batch_ct = self.total_ct_train // self.total_epochs
        self.ct_remainder = self.total_ct_train % self.total_epochs  # 余数（最后一个Epoch补全）

        # 打印CT数据统计
        print(f"CT - Total training pairs: {self.total_ct_train}, "
              f"Batch per epoch: {self.batch_ct}, "
              f"Remainder: {self.ct_remainder}")

    def _init_MR(self, id):
        """修改：保存完整MRI训练对，计算每个Epoch的采样量（与CT逻辑一致）"""
        random.seed(42)
        inputs = "/data1/luyang/data/extracted_top50_samples/MR_LQ"
        targets = "/data1/luyang/data/extracted_top50_samples/MR"

        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))
        if len(lr_files) != len(hr_files):
            print(f"警告: MR文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        split_index = int(len(paired_files) * 0.7)
        self.mr_train_pairs = paired_files[:split_index]
        self.mr_de_type_id = id
        self.total_mr_train = len(self.mr_train_pairs)

        self.batch_mr = self.total_mr_train // self.total_epochs
        self.mr_remainder = self.total_mr_train % self.total_epochs

        print(f"MR - Total training pairs: {self.total_mr_train}, "
              f"Batch per epoch: {self.batch_mr}, "
              f"Remainder: {self.mr_remainder}")

    def _init_synllie(self, id):
        # 设置随机种子确保可复现性
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/Endoscopy_dark"
        targets = "/data1/luyang/data/extracted_top50_samples/Endoscopy"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集
        split_index = int(len(paired_files) * 0.7)
        train_pairs = paired_files[:split_index]

        # 提取训练集文件路径
        train_lr = [pair[0] for pair in train_pairs]
        train_hr = [pair[1] for pair in train_pairs]

        self.synllie_lr = [{"img": x, "de_type": id} for x in train_lr]
        self.synllie_hr = [{"img": x, "de_type": id} for x in train_hr]

        self.synllie_counter = 0
        print("Total Endoscopy training pairs : {}".format(len(self.synllie_lr)))
        print("Repeated Dataset length : {}".format(len(self.synllie_hr)))

    def _init_deblur(self, id):
        """ Initialize the GoPro training dataset with 7:3 split """
        # 设置随机种子确保可复现性
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/Fundus_spot_light"
        targets = "/data1/luyang/data/extracted_top50_samples/Fundus"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集
        split_index = int(len(paired_files) * 0.7)
        train_pairs = paired_files[:split_index]

        # 提取训练集文件路径
        train_lr = [pair[0] for pair in train_pairs]
        train_hr = [pair[1] for pair in train_pairs]

        self.deblur_lr = [{"img": x, "de_type": id} for x in train_lr]
        self.deblur_hr = [{"img": x, "de_type": id} for x in train_hr]

        self.deblur_counter = 0
        print("Total Fundus training pairs : {}".format(len(self.deblur_lr)))
        print("Repeated Dataset length : {}".format(len(self.deblur_hr)))

    def _init_derain(self, id):
        """ Initialize the deraining dataset with 7:3 split """
        # 设置随机种子确保可复现性
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/PET_denoised"
        targets = "/data1/luyang/data/extracted_top50_samples/PET"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集
        split_index = int(len(paired_files) * 0.7)
        train_pairs = paired_files[:split_index]

        # 提取训练集文件路径
        train_lr = [pair[0] for pair in train_pairs]
        train_hr = [pair[1] for pair in train_pairs]

        self.derain_lr = [{"img": x, "de_type": id} for x in train_lr]
        self.derain_hr = [{"img": x, "de_type": id} for x in train_hr]

        self.derain_counter = 0
        print("Total PET training pairs : {}".format(len(self.derain_lr)))
        print("Repeated Dataset length : {}".format(len(self.derain_hr)))
    def _init_dehaze(self, id):
        """ Initialize the deraining dataset with 7:3 split """
        # 设置随机种子确保可复现性
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/Ultrasound_sound_artifacts"
        targets = "/data1/luyang/data/extracted_top50_samples/Ultrasound"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集
        split_index = int(len(paired_files) * 0.7)
        train_pairs = paired_files[:split_index]

        # 提取训练集文件路径
        train_lr = [pair[0] for pair in train_pairs]
        train_hr = [pair[1] for pair in train_pairs]

        self.dehaze_lr = [{"img": x, "de_type": id} for x in train_lr]
        self.dehaze_hr = [{"img": x, "de_type": id} for x in train_hr]

        self.dehaze_counter = 0
        print("Total Ultrasound training pairs : {}".format(len(self.dehaze_lr)))
        print("Repeated Dataset length : {}".format(len(self.dehaze_hr)))
    # def _init_dehaze(self, id):
    #     inputs = self.args.data_file_dir + "/dehazing/RESIDE/"
    #     targets = self.args.data_file_dir + "/dehazing/RESIDE/clear"
    #
    #     self.dehaze_lr = []
    #     for part in ["part1", "part2", "part3", "part4"]:
    #         self.dehaze_lr += [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + part + "/*.jpg"))]
    #
    #     self.dehaze_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.jpg"))]
    #
    #     self.dehaze_counter = 0
    #     print("Total Dehaze training pairs : {}".format(len(self.dehaze_lr)))
    #     self.dehaze_lr = self.dehaze_lr
    #     self.dehaze_hr = self.dehaze_hr
    #     print("Repeated Dataset length : {}".format(len(self.dehaze_lr)))
    def _init_denoise(self, id):
        if 'X-ray' in self.de_type:
            random.seed(42)

            # 原始文件夹路径
            inputs = "/data1/luyang/data/extracted_top50_samples/X_ray_blur"
            targets = "/data1/luyang/data/extracted_top50_samples/X_ray"

            # 获取所有文件列表
            lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
            hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

            # 验证文件数量是否一致
            if len(lr_files) != len(hr_files):
                print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

            # 创建配对列表并打乱顺序
            paired_files = list(zip(lr_files, hr_files))
            random.shuffle(paired_files)

            # 按7:3比例分割数据集
            split_index = int(len(paired_files) * 0.7)
            train_pairs = paired_files[:split_index]

            # 提取训练集文件路径
            train_lr = [pair[0] for pair in train_pairs]
            train_hr = [pair[1] for pair in train_pairs]

            self.denoise_lr = [{"img": x, "de_type": id} for x in train_lr]
            self.denoise_hr = [{"img": x, "de_type": id} for x in train_hr]

            self.denoise_counter = 0
            print("Total X_ray training pairs : {}".format(len(self.denoise_lr)))
            print("Repeated Dataset length : {}".format(len(self.denoise_hr)))
        # inputs = self.args.data_file_dir + "/denoising"
        #
        # clean = []
        # for dataset in ["WaterlooED", "BSD400"]:
        #     if dataset == "WaterlooED":
        #         ext = "bmp"
        #     else:
        #         ext = "jpg"
        #     clean += [x for x in sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))]
        #
        # if 'denoise_15' in self.de_type:
        #     self.s15_ids = [{"img": x, "de_type": self.de_dict['denoise_15']} for x in clean]
        #     self.s15_ids = self.s15_ids * 3
        #     random.shuffle(self.s15_ids)
        #     self.s15_counter = 0
        # if 'denoise_25' in self.de_type:
        #     self.s25_ids = [{"img": x, "de_type": self.de_dict['denoise_25']} for x in clean]
        #     self.s25_ids = self.s25_ids * 3
        #     random.shuffle(self.s25_ids)
        #     self.s25_counter = 0
        # if 'denoise_50' in self.de_type:
        #     self.s50_ids = [{"img": x, "de_type": self.de_dict['denoise_50']} for x in clean]
        #     self.s50_ids = self.s50_ids * 3
        #     random.shuffle(self.s50_ids)
        #     self.s50_counter = 0
        #
        # self.num_clean = len(clean)
        # print("Total Denoise Ids : {}".format(self.num_clean))
    # def _init_clean(self, id):
    #     inputs = self.args.data_file_dir + "/denoising"
    #
    #     clean = []
    #     for dataset in ["WaterlooED", "BSD400"]:
    #         if dataset == "WaterlooED":
    #             ext = "bmp"
    #         else:
    #             ext = "jpg"
    #         clean += [x for x in sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))]
    #
    #     if 'denoise_15' in self.de_type:
    #         self.s15_ids = [{"img": x, "de_type": self.de_dict['denoise_15']} for x in clean]
    #         self.s15_ids = self.s15_ids * 3
    #         random.shuffle(self.s15_ids)
    #         self.s15_counter = 0
    #     if 'denoise_25' in self.de_type:
    #         self.s25_ids = [{"img": x, "de_type": self.de_dict['denoise_25']} for x in clean]
    #         self.s25_ids = self.s25_ids * 3
    #         random.shuffle(self.s25_ids)
    #         self.s25_counter = 0
    #     if 'denoise_50' in self.de_type:
    #         self.s50_ids = [{"img": x, "de_type": self.de_dict['denoise_50']} for x in clean]
    #         self.s50_ids = self.s50_ids * 3
    #         random.shuffle(self.s50_ids)
    #         self.s50_counter = 0
    #
    #     self.num_clean = len(clean)
    #     print("Total Denoise Ids : {}".format(self.num_clean))

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[:2]
        # print(H, W)

        # 如果图像太小，先进行填充或缩放
        if H < self.args.patch_size or W < self.args.patch_size:
            # # 方法 1：填充（padding）到至少 patch_size
            # pad_h = max(0, self.args.patch_size - H)
            # pad_w = max(0, self.args.patch_size - W)
            # img_1 = np.pad(img_1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            # img_2 = np.pad(img_2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            # H, W = img_1.shape[:2]  # 更新尺寸

            # 方法 2：缩放（resize）到至少 patch_size
            scale = max(self.args.patch_size / H, self.args.patch_size / W)
            new_H, new_W = int(H * scale), int(W * scale)
            img_1 = cv2.resize(img_1, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            H, W = new_H, new_W

        # 随机选择裁剪位置
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        # 裁剪 patch
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/clear"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + suffix
        return nonhazy_name


class IRBenchmarks(Dataset):
    def __init__(self, args):
        super(IRBenchmarks, self).__init__()

        self.args = args
        self.benchmarks = args.benchmarks
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}

        self.toTensor = ToTensor()

        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)

        self._init_lr()

    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]

        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks or "denoise_100" in self.benchmarks or "denoise_75" in self.benchmarks:
            sigma = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_noise(hr, sigma)
        else:
            hr_sample = self.hr[idx]
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)

        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        return [lr_sample["img"], de_id], lr, hr

    def __len__(self):
        return len(self.lr)

    def _init_lr(self):
        # print(self.de_type)
        # print("Loading IRBenchmarks dataset...")
        # if 'lolv1' in self.benchmarks:
        #     self._init_synllie(id=self.de_dict['synllie'])
        # if 'gopro' in self.benchmarks:
        #     self._init_deblurring("GoPro", id=self.de_dict['deblur'])
        # if 'derain' in self.benchmarks:
        #     self._init_derain(id=self.de_dict['derain'])
        # if 'dehaze' in self.benchmarks:
        #     self._init_dehaze(id=self.de_dict['dehaze'])
        if 'Endoscopy' in self.de_type:
            self._init_synllie(id=self.de_dict['Endoscopy'])
        if 'Fundus' in self.de_type:
            self._init_deblurring(id=self.de_dict['Fundus'])
        if 'PET' in self.de_type:
            self._init_derain(id=self.de_dict['PET'])
        if 'Ultrasound' in self.de_type:
            self._init_dehaze(id=self.de_dict['Ultrasound'])
        if 'X-ray' in self.de_type:
            self._init_denoise(id=self.de_dict['X-ray'])
        if 'CT' in self.de_type:
            self._init_CT(id=self.de_dict['CT'])  # 关键：CT保存完整训练对
        if 'MR' in self.de_type:
            self._init_MR(id=self.de_dict['MR'])  # 关键：MRI保存完整训练对
        # if 'denoise_15' in self.benchmarks:
        #     self._init_denoise(id=0)
        # if 'denoise_25' in self.benchmarks:
        #     self._init_denoise(id=0)
        # if 'denoise_50' in self.benchmarks:
        #     self._init_denoise(id=0)

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/gt"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + '.png'
        return nonhazy_name

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    ####################################################################################################
    ## DEBLURRING DATASET
    # def _init_deblurring(self, benchmark, id):
    #     inputs = self.args.data_file_dir + f"/deblurring/{benchmark}/test/input/"
    #     targets = self.args.data_file_dir + f"/deblurring/{benchmark}/test/target/"
    #
    #     self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
    #     self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]
    #     print("Total Deblur testing pairs : {}".format(len(self.hr)))

    ####################################################################################################
    # LLIE DATASET
    def _init_CT(self, id):
        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/CT_metal_artifacts"
        targets = "/data1/luyang/data/extracted_top50_samples/CT"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        print("Total CT testing pairs : {}".format(len(self.hr)))
    def _init_MR(self, id):
        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/MR_LQ"
        targets = "/data1/luyang/data/extracted_top50_samples/MR"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        print("Total CT testing pairs : {}".format(len(self.hr)))
    def _init_synllie(self, id):
        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/Endoscopy_dark"
        targets = "/data1/luyang/data/extracted_top50_samples/Endoscopy"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        print("Total LLIE testing pairs : {}".format(len(self.hr)))
    def _init_deblurring(self, id):

        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/Fundus_spot_light"
        targets = "/data1/luyang/data/extracted_top50_samples/Fundus"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        print("Total LLIE testing pairs : {}".format(len(self.hr)))
    def _init_derain(self, id):
        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/PET_denoised"
        targets = "/data1/luyang/data/extracted_top50_samples/PET"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        # print(4333333333333333334444)

        print("Total LLIE testing pairs : {}".format(len(self.hr)))
    def _init_dehaze(self, id):
        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/Ultrasound_sound_artifacts"
        targets = "/data1/luyang/data/extracted_top50_samples/Ultrasound"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        print("Total LLIE testing pairs : {}".format(len(self.hr)))
    def _init_denoise(self, id):
        # 设置随机种子确保与训练集分割一致
        random.seed(42)

        # 原始文件夹路径
        inputs = "/data1/luyang/data/extracted_top50_samples/X_ray_blur"
        targets = "/data1/luyang/data/extracted_top50_samples/X_ray"

        # 获取所有文件列表
        lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
        hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))

        # 验证文件数量是否一致
        if len(lr_files) != len(hr_files):
            print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")

        # 创建配对列表并打乱顺序
        paired_files = list(zip(lr_files, hr_files))
        random.shuffle(paired_files)

        # 按7:3比例分割数据集（取后30%作为测试集）
        split_index = int(len(paired_files) * 0.7)
        test_pairs = paired_files[split_index:]

        # 提取测试集文件路径
        test_lr = [pair[0] for pair in test_pairs]
        test_hr = [pair[1] for pair in test_pairs]

        # 创建测试集数据
        self.lr = [{"img": x, "de_type": id} for x in test_lr]
        self.hr = [{"img": x, "de_type": id} for x in test_hr]

        print("Total LLIE testing pairs : {}".format(len(self.hr)))
    # def _init_synllie(self, id):
    #     # 设置随机种子确保与训练集分割一致
    #     random.seed(42)
    #
    #     # 原始文件夹路径
    #     inputs = "/data1/luyang/data/extracted_top50_samples/Endoscopy_dark"
    #     targets = "/data1/luyang/data/extracted_top50_samples/Endoscopy"
    #
    #     # 获取所有文件列表
    #     lr_files = sorted(glob.glob(os.path.join(inputs, "*.png")))
    #     hr_files = sorted(glob.glob(os.path.join(targets, "*.png")))
    #
    #     # 验证文件数量是否一致
    #     if len(lr_files) != len(hr_files):
    #         print(f"警告: 文件数量不匹配 (LR: {len(lr_files)}, HR: {len(hr_files)})")
    #
    #     # 创建配对列表并打乱顺序
    #     paired_files = list(zip(lr_files, hr_files))
    #     random.shuffle(paired_files)
    #
    #     # 按7:3比例分割数据集（取后30%作为测试集）
    #     split_index = int(len(paired_files) * 0.7)
    #     test_pairs = paired_files[split_index:]
    #
    #     # 提取测试集文件路径
    #     test_lr = [pair[0] for pair in test_pairs]
    #     test_hr = [pair[1] for pair in test_pairs]
    #
    #     # 创建测试集数据
    #     self.lr = [{"img": x, "de_type": id} for x in test_lr]
    #     self.hr = [{"img": x, "de_type": id} for x in test_hr]
    #
    #     print("Total LLIE testing pairs : {}".format(len(self.hr)))
    ####################################################################################################
    ## DERAINING DATASET
    # def _init_derain(self, id):
    #     inputs = self.args.data_file_dir + "/deraining/Rain100L/rainy"
    #     targets = self.args.data_file_dir + "/deraining/Rain100L/gt"
    #
    #     self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
    #     self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]
    #
    #     print("Total Derain testing pairs : {}".format(len(self.hr)))

    ####################################################################################################
    ## DEHAZING DATASET
    # def _init_dehaze(self, id):
    #     inputs = self.args.data_file_dir + "/dehazing/SOTS/outdoor/hazy"
    #     targets = self.args.data_file_dir + "/dehazing/SOTS/outdoor/gt"
    #
    #     self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
    #
    #     self.hr = []
    #     for sample in self.lr:
    #         hazy_name = sample["img"]
    #         clean_name = self._get_nonhazy_name(hazy_name)
    #         self.hr.append({"img": clean_name, "de_type": id})
    #     print("Total Dehazing testing pairs : {}".format(len(self.hr)))

    ####################################################################################################
    ## DENOISING DATASET
    # def _init_denoise(self, id):
    #     inputs = self.args.data_file_dir + "/denoising/cBSD68/original_png"
    #
    #     clean = [x for x in sorted(glob.glob(inputs + "/*.png"))]
    #
    #     self.lr = [{"img": x, "de_type": id} for x in clean]
    #     self.hr = [{"img": x, "de_type": id} for x in clean]
    #     print("Total Denoise testing pairs : {}".format(len(self.lr)))
