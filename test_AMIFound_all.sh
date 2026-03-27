# 方法 2.1：直接加载 conda.sh（推荐）
source /opt/conda/etc/profile.d/conda.sh  # 根据你的 Conda 安装路径调整

# 方法 2.2：或者加载 ~/.bashrc（确保其中包含 Conda 初始化代码）

# 激活环境
conda activate AMIFound

# 运行你的命令
cd /caoluyang/code/AMIFound-main

python src/test_all_patch.py --model AMIFound --benchmarks synllie --checkpoint_id AMIFound_large --de_type synllie --save_results

python src/test_all_patch.py --model AMIFound --benchmarks derain --checkpoint_id AMIFound_large --de_type derain --save_results

python src/test_all_patch.py --model AMIFound --benchmarks dehaze --checkpoint_id AMIFound_large --de_type dehaze --save_results

python src/test_all_patch.py --model AMIFound --benchmarks denoise --checkpoint_id AMIFound_large --de_type denoise --save_results

python src/test_all_patch.py --model AMIFound --benchmarks deblur --checkpoint_id AMIFound_large --de_type deblur --save_results

#python src/test_all.py --model AMIFound --benchmarks MR --checkpoint_id dual_moe --de_type MR --save_results
#
#python src/test_all.py --model AMIFound --benchmarks CT --checkpoint_id dual_moe --de_type CT --save_results
