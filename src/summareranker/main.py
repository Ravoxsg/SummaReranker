# Train a supervised-reranker.

import argparse
import torch
import torch.nn as nn
import datasets
import time

import sys

sys.path.append("/data/mathieu/SummaReranker/src/")

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer, TrainingArguments, default_data_collator
from torch.utils.data.dataloader import DataLoader
from transformers.file_utils import is_datasets_available

from common.utils import seed_everything, check_scores
from common.data_scored import load_data
from utils import *
from dataset import MultitaskRerankingDatasetTrain
from training_utils import *
from model import ModelMultitaskBinary



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--fp16', type=bool, default=True)
parser.add_argument('--deepspeed', type=str, default=None) # "ds_config.json"
parser.add_argument('--sharded_ddp', type=str, default="simple") # ["", "simple"]
parser.add_argument("--local_rank", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")

# data
parser.add_argument('--dataset', type=str, default = "reddit",
                    choices= ["cnndm", "xsum", "reddit"])
parser.add_argument('--generation_methods_str', type=str, default = "diverse_beam_search")
parser.add_argument('--scoring_methods_str', type=str, default = "rouge_1+rouge_2+rouge_l")
parser.add_argument('--sep_symbol', type=str, default="[SEP]")
# train
parser.add_argument('--train_datasets', type=str, default=["first_half_train_shuffled", "second_half_train_shuffled"])
parser.add_argument('--max_train_size', type=int, default=1000000)
# val
parser.add_argument('--val_dataset', type=str, default="val")
parser.add_argument('--max_val_size', type=int, default=10000)
# test
parser.add_argument('--test_dataset', type=str, default="test")
parser.add_argument('--max_test_size', type=int, default=10000)

# base model
parser.add_argument('--base_model_type', type=str, default="pegasus",
                    choices = ["pegasus", "bart"])
parser.add_argument('--num_beams', type=int, default=15)

# model
# candidate selection
parser.add_argument('--filter_out_duplicates', type=bool, default=True)
parser.add_argument('--prune_candidates', type=bool, default=True)
parser.add_argument('--sampling_strat', type=str, default="bottom",
                    choices=["random", "bottom"])
parser.add_argument('--n_positives', type=int, default=1)
parser.add_argument('--n_negatives', type=int, default=1)
parser.add_argument('--max_n_candidates', type=int, default=2)
parser.add_argument('--sharp_pos', type=bool, default=False)
# encoder
parser.add_argument('--model', type=str, default="roberta-large")
parser.add_argument('--model_type', type=str, default="roberta",
                    choices=["bert", "roberta"])
parser.add_argument('--cache_dir', type=str, default="../../../hf_models/roberta-large/")
parser.add_argument('--hidden_size', type=int, default=1024) # 768 / 1024
parser.add_argument('--non_linear_repres', type=bool, default=True)
# tackle source length encoding
parser.add_argument('--separate_source_encoding', type=bool, default=False)
# shared bottom
parser.add_argument('--use_shared_bottom', type=bool, default=True)
parser.add_argument('--bottom_hidden_size', type=int, default=1024)
# experts
parser.add_argument('--num_experts', type=int, default=6)
# parser.add_argument('--noisy_gating', type=bool, default=True)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--use_aux_loss', type=bool, default=False)
parser.add_argument('--expert_hidden_size', type=int, default=1024)
# tower
parser.add_argument('--tower_hidden_size', type=int, default=1024)

# optimization
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--shuffle_train', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--adafactor', type=bool, default=True)
parser.add_argument('--train_bs', type=int, default=1)
parser.add_argument('--inference_bs', type=int, default=60)
parser.add_argument('--gradient_accumulation_steps', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--gradient_clipping', type=float, default=10e10)
parser.add_argument('--scheduler', type=str, default="linear",
                    choices=["constant", "linear"])
parser.add_argument('--warmup_ratio', type=float, default=0.05)

# evaluation
parser.add_argument('--eval_epoch_0', type=bool, default=True)
parser.add_argument('--evaluation_strategy', type=str, default="steps")
parser.add_argument('--n_checkpoints_to_save', type=int, default=2)
parser.add_argument('--metric_for_best_model', type=str, default="overall_sum",
                    choices=["prediction_sum", "overall_sum"])

# export
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--save_model_path', type=str, default="saved_models/reddit/model_1")

args = parser.parse_args()
args.generation_methods = args.generation_methods_str.split("+")
args.scoring_methods = args.scoring_methods_str.split("+")
args.n_tasks = len(args.scoring_methods)

dataset_names = ["cnndm", "xsum", "reddit"]
highlights = [True, False, False]
train_sizes = [[143000, 144113], [102000, 102045], [17000, 16704]]
val_sizes = [13368, 11332, 4213]
test_sizes = [11490, 11334, 4222]

pegasus_train_model_names = [
    ["pegasus_cnndm_second_half_shuffled_1", "pegasus_cnndm_first_half_shuffled_1"],
    ["pegasus_xsum_second_half_shuffled_1", "pegasus_xsum_first_half_shuffled_1"],
    ["pegasus_reddit_second_half_shuffled_1", "pegasus_reddit_first_half_shuffled_1"]
]
bart_train_model_names = [
    ["bart_cnndm_second_half_shuffled_1", "bart_cnndm_first_half_shuffled_1"],
    ["bart_xsum_second_half_shuffled_1", "bart_xsum_first_half_shuffled_1"],
    ["bart_reddit_second_half_shuffled_2", "bart_reddit_first_half_shuffled_2"]
]
pegasus_model_names = [
    "pegasus_cnndm", "pegasus_xsum", "pegasus_reddit_train_1"
]
bart_model_names = [
    "bart_cnndm", "bart_xsum", "bart_reddit"
]
max_lengths = [384, 448, 448]
max_summary_lengths = [128, 64, 64]
eval_every = [500, 500, 100]
clean_ns = [True, False, False]

idx = dataset_names.index(args.dataset)

args.highlights = highlights[idx]
args.train_sizes = train_sizes[idx]
args.val_size = val_sizes[idx]
args.test_size = test_sizes[idx]
if args.base_model_type == "pegasus":
    args.train_model_names = pegasus_train_model_names[idx]
    args.model_name = pegasus_model_names[idx]
elif args.base_model_type == "bart":
    args.train_model_names = bart_train_model_names[idx]
    args.model_name = bart_model_names[idx]
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
args.eval_every = eval_every[idx]
args.clean_n = clean_ns[idx]

print("*" * 50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("Using device: {}".format(device))

    # tokenizer
    tokenizer = build_tokenizer(args)

    # data & datasets
    datasets = []
    for x in [(args.train_datasets, args.train_sizes), (args.val_dataset, args.val_size), (args.test_dataset, args.test_size)]:
        set, size = x
        # data
        train = set == args.train_datasets
        texts, summaries, scored_summaries = load_data(set, size, args, individual_txt=args.highlights, train=train)
        print("loaded new data!", len(texts), len(summaries), len(scored_summaries), len(scored_summaries[0]),
              len(scored_summaries[0][0]), len(scored_summaries[0][1]), len(scored_summaries[0][1][0]))
        # dataset
        mode = "train"
        if not (train):
            mode = "val"
        dataset = MultitaskRerankingDatasetTrain(mode, tokenizer, texts, scored_summaries, summaries, args)
        datasets.append(dataset)
        print("There are {} {} batches".format(int(len(dataset.texts) / args.train_bs), set))
    train_dataset = datasets[0]
    train_dataset.texts = train_dataset.texts[:args.max_train_size]
    train_dataset.scored_summaries = train_dataset.scored_summaries[:args.max_train_size]
    train_dataset.labels = train_dataset.labels[:args.max_train_size]
    val_dataset = datasets[1]
    val_dataset.texts = val_dataset.texts[:args.max_val_size]
    val_dataset.scored_summaries = val_dataset.scored_summaries[:args.max_val_size]
    val_dataset.labels = val_dataset.labels[:args.max_val_size]
    test_dataset = datasets[2]
    test_dataset.texts = test_dataset.texts[:args.max_test_size]
    test_dataset.scored_summaries = test_dataset.scored_summaries[:args.max_test_size]
    test_dataset.labels = test_dataset.labels[:args.max_test_size]

    print(train_dataset.texts[0])
    print("*" * 30)
    print(val_dataset.texts[0])
    print("*" * 30)
    print(test_dataset.texts[0])

    # check oracle
    m_train_score = check_scores(train_dataset)
    m_val_score = check_scores(val_dataset)
    m_test_score = check_scores(test_dataset)
    print("\nOracle - train: {:.4f}, val: {:.4f}, test: {:.4f}".format(m_train_score, m_val_score, m_test_score))

    # model
    pretrained_model = build_model(args)
    model = ModelMultitaskBinary(pretrained_model, tokenizer, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)

    train_args = TrainingArguments(
        output_dir=args.save_model_path,  # will be changed
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_every,
        save_total_limit=args.n_checkpoints_to_save,
        num_train_epochs=args.n_epochs,
        adafactor=args.adafactor,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.inference_bs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.wd,
        max_grad_norm=args.gradient_clipping,
        logging_strategy="no",
        save_strategy=args.evaluation_strategy,
        save_steps=args.eval_every,
        metric_for_best_model=args.metric_for_best_model,
        fp16=args.fp16,
        load_best_model_at_end=True,
        greater_is_better=True,
        disable_tqdm=False,
        deepspeed=args.deepspeed,
        sharded_ddp=args.sharded_ddp,
        local_rank=args.local_rank
    )

    data_collator = default_data_collator

    trainer = CustomTrainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    if args.eval_epoch_0:
        results = trainer.evaluate()
        print("*" * 50, "Init VAL results:")
        print(results)
        model.moe.display_tasks_probs()

    # training loop
    if args.train:
        trainer.train()
        model.display_training_labels()
    else:
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model_path))
            print("Loaded the model weights!", args.load_model_path)

    # validate with the best model
    results = trainer.evaluate()
    print("\n", "*" * 50, "BEST VAL RESULTS")
    print(results)
    model.moe.display_tasks_probs()

    # test results
    test_results = trainer.predict(test_dataset)
    print("\n", "*" * 50, "TEST RESULTS:")
    print(test_results[2])
    model.moe.display_tasks_probs()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        mode = inputs["mode"]
        text_and_summaries_ids = inputs["text_and_summaries_input_ids"]
        text_and_summaries_mask = inputs["text_and_summaries_attn_mask"]
        scores = inputs["scores"]

        outputs = model(mode, text_and_summaries_ids, text_and_summaries_mask, scores)

        loss = outputs["loss"]
        output = torch.zeros(2 + 3 * args.n_tasks + 2).float().to(loss.device)
        output[0] = loss
        output[1] = outputs["loss_nce"]
        for j in range(args.n_tasks):
            output[2 + j * 3] = outputs["accuracy_{}".format(args.scoring_methods[j])]
            output[3 + j * 3] = outputs["rank_{}".format(args.scoring_methods[j])]
            output[4 + j * 3] = outputs["prediction_{}".format(args.scoring_methods[j])]
        output[-2] = outputs["prediction_sum"]
        output[-1] = outputs["overall_sum"]

        return (loss, output) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    # with autocast():
                    outputs = model(**inputs)
                else:
                    text_inputs_ids = inputs["text_inputs_ids"]
                    text_attention_mask = inputs["text_attention_mask"]
                    text_inputs = {
                        "input_ids": text_inputs_ids,
                        "attention_mask": text_attention_mask
                    }
                    outputs = model(**text_inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=train_dataset.args.shuffle_train,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    loss_nce = np.mean([preds[i] for i in range(0, len(preds), 1 + 3 * args.n_tasks + 2)])
    result = {
        "loss_nce": loss_nce
    }
    for j in range(args.n_tasks):
        accuracy_arr = [preds[i] for i in range(1 + j * 3, len(preds), 1 + 3 * args.n_tasks + 2)]
        accuracy = np.mean(accuracy_arr)
        rank_arr = [preds[i] for i in range(2 + j * 3, len(preds), 1 + 3 * args.n_tasks + 2)]
        rank = np.mean(rank_arr)
        prediction_arr = [preds[i] for i in range(3 + j * 3, len(preds), 1 + 3 * args.n_tasks + 2)]
        prediction = np.mean(prediction_arr)
        print("Task {}, # pred batches: {}".format(j + 1, len(accuracy_arr)))
        result["accuracy_{}".format(args.scoring_methods[j])] = accuracy
        result["rank_{}".format(args.scoring_methods[j])] = rank
        result["prediction_{}".format(args.scoring_methods[j])] = prediction
    prediction_sum = np.mean([preds[i] for i in range(1 + 3 * args.n_tasks, len(preds), 1 + 3 * args.n_tasks + 2)])
    result["prediction_sum"] = prediction_sum
    overall_sum = np.mean([preds[i] for i in range(1 + 3 * args.n_tasks + 1, len(preds), 1 + 3 * args.n_tasks + 2)])
    result["overall_sum"] = overall_sum

    return result



if __name__ == '__main__':
    main(args)
