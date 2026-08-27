source /opt/conda/etc/profile.d/conda.sh
conda activate AMIFound
cd /caoluyang/code/AMIFound-main
python src/test_all_patch.py --model AMIFound --benchmarks Endoscopy --checkpoint_id AMIFound_large --de_type Endoscopy --save_results
python src/test_all_patch.py --model AMIFound --benchmarks PET --checkpoint_id AMIFound_large --de_type PET --save_results
python src/test_all_patch.py --model AMIFound --benchmarks Ultrasound --checkpoint_id AMIFound_large --de_type Ultrasound --save_results
python src/test_all_patch.py --model AMIFound --benchmarks X-ray --checkpoint_id AMIFound_large --de_type X-ray --save_results
python src/test_all_patch.py --model AMIFound --benchmarks Fundus --checkpoint_id AMIFound_large --de_type Fundus --save_results
python src/test_all_patch.py --model AMIFound --benchmarks CT --checkpoint_id AMIFound_large --de_type CT --save_results
python src/test_all_patch.py --model AMIFound --benchmarks MRI --checkpoint_id AMIFound_large --de_type MRI --save_results
