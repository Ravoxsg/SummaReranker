# Supervised fine-tuning of language models.

import argparse
import time
import torch.nn as nn
import sys

sys.path.append("/data/mathieu/SummaReranker/src/")

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, default_data_collator

from common.summary_processing import pre_rouge_processing
from utils import *
from data import *
from dataset import *
from dataset_trainer import *
from transfer_utils import *
from model import FTModel



parser = argparse.ArgumentParser()

# general
parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--cuda', type=bool, default = True)
parser.add_argument('--mp', type=bool, default = False)
parser.add_argument('--debug', type=bool, default = False)
parser.add_argument('--debug_size', type=int, default = 20)
parser.add_argument('--deepspeed', type=str, default = None)  # "ds_config.json"
parser.add_argument('--sharded_ddp', type=str, default = "simple")  # ["", "simple"]
parser.add_argument("--local_rank", type=int, default = 0, help="Local rank. Necessary for using the torch.distributed.launch utility.")

# task
parser.add_argument('--train', type=bool, default = True)

# data
parser.add_argument('--dataset', type=str, default = "reddit",
                    choices=["cnndm", "xsum", "reddit"])
# train
parser.add_argument('--train_dataset', type = str, default = "train")
parser.add_argument('--max_train_size', type=int, default = 1000000)
# val
parser.add_argument('--val_dataset', type = str, default = "val")
parser.add_argument('--max_val_size', type = int, default = 1000000)
# test
parser.add_argument('--test_dataset', type = str, default = "test")
parser.add_argument('--max_test_size', type = int, default = 1000000)

# model
parser.add_argument('--model_type', type=str, default = "pegasus",
                    choices=["pegasus", "bart"])
parser.add_argument('--model', type=str, default = "google/pegasus-large",
                    choices=["google/pegasus-large", "facebook/bart-large"])
parser.add_argument('--hidden_size', type=int, default = 768)
parser.add_argument('--cache_dir', type=str, default = "../../../hf_models/pegasus-large/") # in ["pegasus-large", "bart-large"]
parser.add_argument('--load_model', type=bool, default = False)
parser.add_argument('--load_model_path', type=str, default = "")

# optimization
parser.add_argument('--n_epochs', type=int, default = 15)
parser.add_argument('--inference_bs', type=int, default = 2)
parser.add_argument('--wd', type=float, default = 0)
parser.add_argument('--gradient_clipping', type=float, default = 10e10)
parser.add_argument('--label_smoothing', type=float, default = 0.1)

# generation
parser.add_argument('--repetition_penalty', type = float, default = 1.0)

# evaluation
parser.add_argument('--eval_epoch_0', type = bool, default = True)
parser.add_argument('--evaluation_strategy', type=str, default = "steps")
parser.add_argument('--evaluation_method', type=str, default = "generation",
                    choices=["generation", "loss"])
parser.add_argument('--eval_test', type=bool, default = False)
parser.add_argument('--eval_every', type=int, default=-1)

# summaries
parser.add_argument('--generate_summaries', type=bool, default = False)
parser.add_argument('--stemmer', type=bool, default = True)
parser.add_argument('--show_summaries', type=bool, default = True)
parser.add_argument('--show_summaries_count', type=int, default = 1) # batches

# export
parser.add_argument('--n_checkpoints_to_save', type=int, default = 2)
parser.add_argument('--save_model_path', type=str, default = "ft_saved_models/reddit/pegasus_reddit_train_1")

args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit"]
max_lengths = [1024, 512, 512]
train_sizes = [287113, 204045, 33704]
first_half_train_sizes = [143000, 102000, 17000]
second_half_train_sizes = [144113, 102045, 16704]
val_sizes = [13368, 11332, 4213]
test_sizes = [11490, 11334, 4222]
lrs_pegasus = [5e-5, 1e-4, 1e-4]
lrs_barts = [3e-5, 3e-5, 3e-5]
eval_every_pegasus = [300, 250, 100]
eval_every_bart = [500, 250, 100]
max_summary_lengths = [128, 64, 64]
length_penalties_pegasus = [0.8, 0.8, 0.6]
length_penalties_bart = [0.8, 0.8, 1.0]
no_repeat_ngram_sizes_pegasus = [0, 3, 3]
no_repeat_ngram_sizes_bart = [0, 3, 3]
highlights = [True, False, False]
clean_ns = [True, False, False]

idx = dataset_names.index(args.dataset)
args.data_folder = "../../data/{}".format(args.dataset)
args.max_length = max_lengths[idx]
args.train_size = train_sizes[idx]
args.val_size = val_sizes[idx]
args.test_size = test_sizes[idx]
# optimization
if args.model_type == "pegasus":
    args.fp16 = False
    args.adafactor = True
    args.lr = lrs_pegasus[idx]
    args.train_bs = 2
    args.scheduler = "constant"
    args.warmup_ratio = 0
    args.gradient_accumulation_steps = 128
    if args.eval_every < 0:
        args.eval_every = eval_every_pegasus[idx]
elif args.model_type == "bart":
    args.fp16 = True
    args.adafactor = False
    args.lr = lrs_barts[idx]
    args.train_bs = 4
    args.scheduler = "linear"
    args.warmup_ratio = 0.025
    args.gradient_accumulation_steps = 20
    if args.eval_every < 0:
        args.eval_every = eval_every_bart[idx]
args.max_summary_length = max_summary_lengths[idx]
# summary generation
if args.model_type == "pegasus":
    args.length_penalty = length_penalties_pegasus[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_pegasus[idx]
    args.num_beams = 8
elif args.model_type == "bart":
    args.length_penalty = length_penalties_bart[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_bart[idx]
    args.num_beams = 5
args.highlights = highlights[idx]
args.clean_n = clean_ns[idx]

if args.evaluation_method == "loss":
    args.metric_for_best_model = "loss"
    args.greater_is_better = False
elif args.evaluation_method == "generation":
    args.metric_for_best_model = "mean_r"
    args.greater_is_better = True

if args.debug:
    args.max_val_size = 10
    args.eval_every = 10
    args.gradient_accumulation_steps = 2
    args.n_checkpoints_to_save = 0

print("*" * 50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # data
    train_data = load_data(args.train_dataset, args, individual_txt=False)
    val_data = load_data(args.val_dataset, args, individual_txt=False)
    test_data = load_data(args.test_dataset, args, individual_txt=False)

    # tokenizer
    tokenizer = build_tokenizer(args)

    # datasets
    datasets = []
    for x in [("val", val_data), ("test", test_data), ("train", train_data)]:
        mode, data = x
        texts, summaries = data
        print(len(texts), len(summaries))
        if args.debug:
            texts = texts[:args.debug_size]
            summaries = summaries[:args.debug_size]
        if mode == "train":
            texts = texts[:args.max_train_size]
            summaries = summaries[:args.max_train_size]
            train_dataset = TrainFTDatasetTrainer(mode, tokenizer, texts, summaries, args)
            datasets.append(train_dataset)
            print("There are {} train data points".format(len(texts)))
        else:
            if mode == "val":
                texts = texts[:args.max_val_size]
                summaries = summaries[:args.max_val_size]
            else:
                texts = texts[:args.max_test_size]
                summaries = summaries[:args.max_test_size]
            dataset = InferenceFTDatasetTrainer(mode, tokenizer, texts, summaries, args)
            datasets.append(dataset)
            print("There are {} {} batches".format(int(len(dataset.texts) / args.train_bs), mode))
    train_dataset = datasets[2]
    val_dataset = datasets[0]
    test_dataset = datasets[1]

    # model
    base_model = build_model(args)
    model = FTModel(base_model, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))

    # loading checkpoint
    if args.load_model:
        print("Loading checkpoint: {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    if args.mp:
        if "t5" in args.model_type:
            print("Using model parallelism...")
            model.parallelize()
        else:
            print("Can't do Model Parallelism on that model")
            raise Exception

    train_args = Seq2SeqTrainingArguments(
        output_dir = args.save_model_path,  # will be changed
        overwrite_output_dir = True,
        do_train = True,
        do_eval = True,
        do_predict = False,
        evaluation_strategy = args.evaluation_strategy,
        eval_steps = args.eval_every,
        save_total_limit = args.n_checkpoints_to_save,
        save_steps = args.eval_every,
        num_train_epochs = args.n_epochs,
        adafactor = args.adafactor,
        lr_scheduler_type = args.scheduler,
        warmup_ratio = args.warmup_ratio,
        per_device_train_batch_size = args.train_bs,
        per_device_eval_batch_size = args.inference_bs,
        learning_rate = args.lr,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        weight_decay = args.wd,
        max_grad_norm = args.gradient_clipping,
        label_smoothing_factor = args.label_smoothing,
        logging_strategy = "no",
        save_strategy = args.evaluation_strategy,
        fp16 = args.fp16,
        load_best_model_at_end = True,
        metric_for_best_model = args.metric_for_best_model,
        greater_is_better = args.greater_is_better,
        disable_tqdm = False,
        deepspeed = args.deepspeed,
        sharded_ddp = args.sharded_ddp,
        local_rank = args.local_rank,
    )

    data_collator = default_data_collator

    trainer = CustomTrainer(
        model = model,
        args = train_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        tokenizer = tokenizer,
    )

    if args.eval_epoch_0:
        results = trainer.evaluate()
        print("*" * 50, "EPOCH 0 RESULTS")
        print(results)

    # training loop
    trainer.train()

    # validate with the best model
    results = trainer.evaluate()
    print("*" * 50, "FINAL RESULTS")
    print(results)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        text_inputs_ids = inputs["text_input_ids"]
        text_attention_mask = inputs["text_attention_mask"]
        labels = inputs["labels"]
        outputs = model(text_inputs_ids, text_attention_mask, labels=labels)
        loss_ce = outputs["loss"]
        loss = loss_ce

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        eval_loop = self.evaluation_loop
        metrics = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        #args = self.args
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=args.stemmer)

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        total_size = 0
        all_losses, all_r1s, all_r2s, all_rls = 0, 0, 0, 0

        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            bs = inputs['text_input_ids'].shape[0]
            total_size += bs

            # Prediction step
            if args.evaluation_method == "loss":
                loss, _, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                all_losses += loss.item() * bs
            elif args.evaluation_method == "generation":
                summary_ids = model.pretrained_model.generate(
                    inputs['text_input_ids'].cuda(),
                    attention_mask=inputs["text_attention_mask"].cuda(),
                    use_cache=True,
                    num_beams=args.num_beams,
                    num_return_sequences=1,
                    # min_length = args.min_summary_length,
                    max_length=args.max_summary_length,
                    early_stopping=True,
                    repetition_penalty=args.repetition_penalty,
                    length_penalty=args.length_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size
                )
                summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                labels_ids = inputs["summary_input_ids"]
                labels = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                temp_r1s, temp_r2s, temp_rls = [], [], []
                for j in range(len(summaries)):
                    label = labels[j]
                    summary = summaries[j]
                    summary = pre_rouge_processing(summary, args)
                    rouge_scores = scorer.score(label, summary)
                    r1 = rouge_scores["rouge1"].fmeasure
                    r2 = rouge_scores["rouge2"].fmeasure
                    rl = rouge_scores["rougeLsum"].fmeasure
                    temp_r1s.append(r1)
                    temp_r2s.append(r2)
                    temp_rls.append(rl)
                all_r1s += np.mean(temp_r1s) * bs
                all_r2s += np.mean(temp_r2s) * bs
                all_rls += np.mean(temp_rls) * bs

        metrics = {}
        if args.evaluation_method == "loss":
            metrics["eval_loss"] = all_losses / total_size
        elif args.evaluation_method == "generation":
            metrics["r1"] = 100 * all_r1s / total_size
            metrics["r2"] = 100 * all_r2s / total_size
            metrics["rl"] = 100 * all_rls / total_size
            metrics["eval_mean_r"] = (metrics["r1"] + metrics["r2"] + metrics["rl"]) / 3
        print(metrics)

        return metrics

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
                    with autocast():
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


if __name__ == '__main__':
    main(args)
