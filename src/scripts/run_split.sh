LAYERS=1
LOGDIR=../logs
python ../run_split.py --num_layers=$LAYERS --runs=5 --log_dir=$LOGDIR --dataset=normal --implicit_beta --h 5 50                     --tag=split_normal_l1_mh > split_normal_l1_mh.log
python ../run_split.py --num_layers=$LAYERS --runs=5 --log_dir=$LOGDIR --dataset=normal --implicit_beta --h 5 50 --single_head       --tag=split_normal_l1_sh > split_normal_l1_sh.log
python ../run_split.py --num_layers=$LAYERS --runs=5 --log_dir=$LOGDIR --dataset=normal --implicit_beta --h 5 50 --single_head --cl3 --tag=split_normal_l1_cl3 > split_normal_l1_cl3.log