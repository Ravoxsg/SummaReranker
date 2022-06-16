python main_scores.py \
--dataset reddit \
--val_dataset val \
--generation_method diverse_beam_search \
--model_name pegasus_reddit_train_1 \
--num_candidates 15 \
--label_metric rouge_1 \
--save_scores True \
