# 方法 2.1：直接加载 conda.sh（推荐）
source /opt/conda/etc/profile.d/conda.sh  # 根据你的 Conda 安装路径调整

# 方法 2.2：或者加载 ~/.bashrc（确保其中包含 Conda 初始化代码）
# source ~/.bashrc

# 激活环境
conda activate AMIFound

# 运行你的命令
cd /caoluyang/code/AMIFound-main

python src/train_all_multiexpers2.py --model AMIFound --batch_size 8 --de_type MR CT denoise dehaze derain deblur synllie --trainset standard --num_gpus 4 --loss_type FFT --fft_loss_weight 0.1 --balance_loss_weight 0.01
