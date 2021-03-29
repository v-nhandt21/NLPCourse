import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import seaborn as sns


device="cuda"


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        self.encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        print(text_fea.shape)
        return loss, text_fea


















