# Model trained on 1st training half => infer on 2nd training half
python main_candidate_generation.py \
--dataset reddit \
--val_dataset second_half_train_shuffled \
--model_type pegasus \
--model google/pegasus-large \
--model_name pegasus_reddit_first_half_shuffled_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path ../base_model_finetuning/ft_saved_models/reddit/pegasus_reddit_first_half_shuffled_1/checkpoint-5/pytorch_model.bin \
--inference_bs 2 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

# Model trained on 2nd training half => infer on 1st training half
python main_candidate_generation.py \
--dataset reddit \
--val_dataset first_half_train_shuffled \
--model_type pegasus \
--model google/pegasus-large \
--model_name pegasus_reddit_second_half_shuffled_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path ../base_model_finetuning/ft_saved_models/reddit/pegasus_reddit_second_half_shuffled_1/checkpoint-5/pytorch_model.bin \
--inference_bs 2 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

# Model trained on entire training set => infer on validation set
python main_candidate_generation.py \
--dataset reddit \
--val_dataset val \
--model_type pegasus \
--model google/pegasus-large \
--model_name pegasus_reddit_train_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path ../base_model_finetuning/ft_saved_models/reddit/pegasus_reddit_train_1/checkpoint-5/pytorch_model.bin \
--inference_bs 2 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

# Model trained on entire training set => infer on test set
python main_candidate_generation.py \
--dataset reddit \
--val_dataset test \
--model_type pegasus \
--model google/pegasus-large \
--model_name pegasus_reddit_train_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path ../base_model_finetuning/ft_saved_models/reddit/pegasus_reddit_train_1/checkpoint-5/pytorch_model.bin \
--inference_bs 2 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \
