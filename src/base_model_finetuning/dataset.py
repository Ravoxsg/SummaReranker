import numpy as np
import torch



class TrainFTDataset:
    def __init__(self, mode, tokenizer, texts, summaries, args):
        self.mode = mode
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.args = args

        print("Dataset has {} training points".format(len(self.texts)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]

        text_inputs = self.tokenizer(text, return_tensors = "pt", max_length = self.args.max_length, padding = 'max_length')
        text_inputs["input_ids"] = text_inputs["input_ids"][:,:self.args.max_length]
        text_inputs["attention_mask"] = text_inputs["attention_mask"][:,:self.args.max_length]

        summary = self.summaries[item]

        summary_inputs = self.tokenizer(summary, return_tensors = "pt", max_length=self.args.max_summary_length, padding='max_length')
        summary_inputs["input_ids"] = summary_inputs["input_ids"][:,:self.args.max_summary_length]
        summary_inputs["attention_mask"] = summary_inputs["attention_mask"][:,:self.args.max_summary_length]

        batch = {
            'text': text,
            'text_inputs': text_inputs,
            'summary': summary,
            'summary_inputs': summary_inputs,
        }

        return batch


class InferenceFTDataset:

    def __init__(self, set, tokenizer, texts, summaries, args):

        self.set = set
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.args = args

        print("Dataset has {} points".format(len(self.texts)))

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        summary = self.summaries[item]

        text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_length, padding='max_length')
        text_inputs["input_ids"] = text_inputs["input_ids"][:, :self.args.max_length]
        text_inputs["attention_mask"] = text_inputs["attention_mask"][:, :self.args.max_length]

        summary_inputs = self.tokenizer(summary, return_tensors="pt", max_length=self.args.max_summary_length, padding='max_length')
        summary_inputs["input_ids"] = summary_inputs["input_ids"][:, :self.args.max_summary_length]
        summary_inputs["attention_mask"] = summary_inputs["attention_mask"][:, :self.args.max_summary_length]

        batch = {
            'text': text,
            'text_inputs': text_inputs,
            'summary': summary,
            'summary_inputs': summary_inputs,
        }

        return batch
