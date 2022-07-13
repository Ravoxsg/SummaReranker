import torch

from time import time



class MultitaskRerankingDataset:
    def __init__(self, mode, tokenizer, texts, scored_summaries, labels, args):
        self.mode = mode
        self.tokenizer = tokenizer
        self.texts = texts
        self.scored_summaries = scored_summaries
        self.labels = labels
        self.args = args

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        scored_summaries = self.scored_summaries[item]
        summary_candidates = scored_summaries[0]
        summary_scores = scored_summaries[1]
        for i in range(len(summary_scores)):
            # Re-adjust for BERTScore
            if min(summary_scores[i]) > 0.0 and max(summary_scores[i]) < 1.0:
                for j in range(len(summary_scores[i])):
                    summary_scores[i][j] *= 100
            # Re-adjust for BARTScore
            if min(summary_scores[i]) > -10.0 and max(summary_scores[i]) < 0.0:
                for j in range(len(summary_scores[i])):
                    summary_scores[i][j] *= 30

        text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_length, padding='max_length')
        text_inputs["input_ids"] = text_inputs["input_ids"][:, :self.args.max_length]
        text_inputs["attention_mask"] = text_inputs["attention_mask"][:, :self.args.max_length]
        summary_candidates_inputs = self.tokenizer(summary_candidates, return_tensors="pt", truncation=True, max_length=self.args.max_summary_length, padding='max_length')
        summary_candidates_inputs["input_ids"] = summary_candidates_inputs["input_ids"][:,:self.args.max_summary_length]
        summary_candidates_inputs["attention_mask"] = summary_candidates_inputs["attention_mask"][:,:self.args.max_summary_length]

        text_and_summaries = [self.tokenizer.decode(text_inputs["input_ids"][0], skip_special_tokens=True) + " " + self.args.sep_symbol + " " \
                              + self.tokenizer.decode(summary_candidates_inputs["input_ids"][i], skip_special_tokens=True) for i in range(len(summary_candidates_inputs["input_ids"]))]
        text_and_summaries_inputs = self.tokenizer(text_and_summaries, return_tensors="pt", truncation=True, max_length=self.args.max_length + self.args.max_summary_length, padding='max_length')
        text_and_summaries_inputs["input_ids"] = text_and_summaries_inputs["input_ids"][:, :(self.args.max_length + self.args.max_summary_length)]
        text_and_summaries_inputs["attention_mask"] = text_and_summaries_inputs["attention_mask"][:, :(self.args.max_length + self.args.max_summary_length)]

        scores = torch.cat([torch.tensor(summary_scores[i]).unsqueeze(0) for i in range(len(summary_scores))], 0)
        labels = torch.max(scores, dim = 1)[0]
        mode = torch.tensor([1])
        if self.mode != "train":
            mode = torch.tensor([0])

        batch = {
            "mode": mode,
            "text": text,
            "label": label,
            "text_and_summaries_input_ids": text_and_summaries_inputs["input_ids"],
            "text_and_summaries_attn_mask": text_and_summaries_inputs["attention_mask"],
            "scores": scores,
            "labels": labels
        }

        return batch



class MultitaskRerankingDatasetTrain:
    def __init__(self, mode, tokenizer, texts, scored_summaries, labels, args):
        self.mode = mode
        self.tokenizer = tokenizer
        self.texts = texts
        self.scored_summaries = scored_summaries
        self.labels = labels
        self.args = args

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        scored_summaries = self.scored_summaries[item]
        summary_candidates = scored_summaries[0]
        summary_scores = scored_summaries[1]
        for i in range(len(summary_scores)):
            # re-adjust BERTScore
            if min(summary_scores[i]) > 0.0 and max(summary_scores[i]) < 1.0:
                for j in range(len(summary_scores[i])):
                    summary_scores[i][j] *= 100
            # re-adjust BARTScore
            elif min(summary_scores[i]) > -10.0 and max(summary_scores[i]) < 0.0:
                for j in range(len(summary_scores[i])):
                    summary_scores[i][j] *= 30

        text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_length, padding='max_length')
        text_inputs["input_ids"] = text_inputs["input_ids"][:, :self.args.max_length]
        text_inputs["attention_mask"] = text_inputs["attention_mask"][:, :self.args.max_length]

        summary_candidates_inputs = self.tokenizer(summary_candidates, return_tensors="pt", truncation=True, max_length=self.args.max_summary_length, padding='max_length')
        summary_candidates_inputs["input_ids"] = summary_candidates_inputs["input_ids"][:,:self.args.max_summary_length]
        summary_candidates_inputs["attention_mask"] = summary_candidates_inputs["attention_mask"][:,:self.args.max_summary_length]

        text_and_summaries = [self.tokenizer.decode(text_inputs["input_ids"][0], skip_special_tokens=True) + " " + self.args.sep_symbol + " " \
                              + self.tokenizer.decode(summary_candidates_inputs["input_ids"][i], skip_special_tokens=True) for i in range(len(summary_candidates_inputs["input_ids"]))]
        text_and_summaries_inputs = self.tokenizer(text_and_summaries, return_tensors="pt", truncation=True, max_length=self.args.max_length + self.args.max_summary_length, padding='max_length')
        text_and_summaries_inputs["input_ids"] = text_and_summaries_inputs["input_ids"][:, :(self.args.max_length + self.args.max_summary_length)]
        text_and_summaries_inputs["attention_mask"] = text_and_summaries_inputs["attention_mask"][:, :(self.args.max_length + self.args.max_summary_length)]

        scores = torch.cat([torch.tensor(summary_scores[i]).unsqueeze(0) for i in range(len(summary_scores))], 0)
        labels = torch.max(scores, dim = 1)[0]
        mode = torch.tensor([1])
        if self.mode != "train":
            mode = torch.tensor([0])

        batch = {
            "mode": mode,
            "text_and_summaries_input_ids": text_and_summaries_inputs["input_ids"],
            "text_and_summaries_attn_mask": text_and_summaries_inputs["attention_mask"],
            "scores": scores,
            "labels": labels
        }

        return batch
