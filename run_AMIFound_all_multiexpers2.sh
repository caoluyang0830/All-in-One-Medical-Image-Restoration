source /opt/conda/etc/profile.d/conda.sh
conda activate AMIFound
cd /caoluyang/code/AMIFound-main
python src/train_all_multiexpers2.py --model AMIFound --batch_size 8 --de_type MR CT X-ray Ultrasound PET Fundus Endoscopy --trainset standard --num_gpus 4 --loss_type FFT --fft_loss_weight 0.1 --balance_loss_weight 0.01
