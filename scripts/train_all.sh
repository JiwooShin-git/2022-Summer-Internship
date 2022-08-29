# python3.7 train_student.py --model_t resnet110 --model_s resnet110_s --use_depth True --is_wandb True --optimizer sgd --remain_block_num 9 --distill kd --lambda1 0.1 --lambda2 0.1 --alpha 0.9 --beta 0 --gamma 0.1 --random_seed_s 2 --random_seed_t 1 --wandb_project resnet110_resnet110_s_9_blocks_depth_True_sigmoid_3

python3.7 train_student.py --model_t resnet110 --model_s resnet110_s --use_depth False --is_wandb True --optimizer sgd --remain_block_num 15 --distill kd --lambda1 0.3 --lambda2 0.3 --alpha 0.9 --beta 0 --gamma 0.1 --random_seed_s 2 --random_seed_t 1 --wandb_project resnet110_resnet110_s_15_blocks_depth_False_sigmoid_3

python3.7 train_student.py --model_t resnet110 --model_s resnet110_s --use_depth True --is_wandb True --optimizer sgd --remain_block_num 21 --distill kd --lambda1 0.1 --lambda2 0.1 --alpha 0.9 --beta 0 --gamma 0.1 --random_seed_s 2 --random_seed_t 1 --wandb_project resnet110_resnet110_s_21_blocks_depth_True_sigmoid_3
