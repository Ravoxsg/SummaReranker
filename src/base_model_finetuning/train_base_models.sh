# split the training set into 2 halves
python build_training_splits.py \
--dataset reddit \

# fine-tune model on the 1st half of the training set
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2000 main_trainer.py \
--dataset reddit \
--train_dataset first_half_train_shuffled \
--model_type pegasus \
--save_model_path ft_saved_models/reddit/pegasus_reddit_first_half_shuffled_1 \

# fine-tune model on the 2nd half of the training set
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2000 main_trainer.py \
--dataset reddit \
--train_dataset second_half_train_shuffled \
--model_type pegasus \
--save_model_path ft_saved_models/reddit/pegasus_reddit_second_half_shuffled_1 \

# fine-tune model on the entire training set (for Reddit)
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2000 main_trainer.py \
--dataset reddit \
--train_dataset train \
--model_type pegasus \
--save_model_path ft_saved_models/reddit/pegasus_reddit_train_1 \
