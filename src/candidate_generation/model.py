import torch
import torch.nn as nn
import numpy as np

from time import time



class FTModel(nn.Module):

    def __init__(self, pretrained_model, args):

        super(FTModel, self).__init__()

        self.pretrained_model = pretrained_model
        self.args = args

    def forward(self, input_ids, attention_mask, labels):

        output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True
        )

        loss_ce = output["loss"]
        loss_ce = torch.nan_to_num(loss_ce)
        outputs = {
            "loss": loss_ce,
            "loss_ce": loss_ce,
        }

        return outputs