import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torchinfo import summary
import pandas as pd
import tensorflow as tf
device="cuda"
class CNN(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers,embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.embedding.weight=nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))

        self.seq_len = 150
        self.fc_out = nn.Linear(300, n_output)
        
        self.conv = nn.Conv1d(in_channels=300, out_channels=100,kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=100,kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=300, out_channels=100,kernel_size=7, padding=3)
        
        self.pooling = nn.MaxPool1d(150)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.3)
        
    def forward (self, input_words):                                                        # => (batch size, sent len)

        batch = input_words.shape[0]
        seq_lenght= input_words.size(1)
        
        embedded_words = self.embedding(input_words)                                        # => (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(0,2,1)                                      # => (batch_size,  n_embed,seq_length)
        
        x = self.conv(embedded_words)           #[512, 100, 150]
        x = self.relu(x)
        x = self.pooling(x)                     #[512, 100, 1]
        
        x1 = self.conv1(embedded_words)
        x1 = self.relu(x1)
        x1 = self.pooling(x1)
        
        x2 = self.conv2(embedded_words)
        x2 = self.relu(x2)
        x2 = self.pooling(x2)
        
        
        out = self.fc_out(torch.cat(    (x.squeeze(), x1.squeeze(), x2.squeeze() )  , 1 )            )
        
        sig = self.sigmoid(out)
        return sig, x