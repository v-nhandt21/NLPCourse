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
class NetworkRNN(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        self.rnn = nn.RNN(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
        
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.6)
        
        
    def forward (self, input_words):
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        rnn_out, hidden = self.rnn(embedded_words)         # (batch_size, seq_length, n_hidden)
        rnn_out = self.dropout(rnn_out)
        out = self.fc(rnn_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, hidden

class NetworkRNN_reimplement(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
#         self.Wi = nn.Parameter(torch.randn( (n_embed,hidden_node), requires_grad=True, dtype=torch.float))
#         self.Wh = nn.Parameter(torch.randn( (hidden_node,hidden_node) , requires_grad=True, dtype=torch.float))
        
        self.hidden_node = hidden_node
        self.embedding = nn.Embedding(n_vocab, n_embed)
#         self.rnn = nn.RNN(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
#         self.rnn_cell = nn.RNNCell(n_embed, hidden_node)
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

        self.linear_hidden = nn.Linear(n_hidden, n_hidden)
        self.linear_input = nn.Linear(n_embed, n_hidden)
        
    def forward (self, input_words):                    # => (batch size, sent len)
        
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)   #  (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)  # batch-node
        
        for i in range(input_words.size(1)):           #for i in seq_length

            A = self.linear_hidden(hidden)
            B = self.linear_input(embedded_words[i])
            C =  A.add(B)
            hidden = self.sigmoid(C)
        
        out = self.fc(hidden)
        
        sig = self.sigmoid(out)
        return sig, hidden

class NetworkGRU(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        self.gru = nn.GRU(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
        
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.8)
        
        
    def forward (self, input_words):
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        gru_out, hidden = self.gru(embedded_words)         # (batch_size, seq_length, n_hidden)
        gru_out = self.dropout(gru_out)
        
        out = self.fc(gru_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, hidden

class NetworkGRU_reimplement(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        self.hidden_node = hidden_node
        n_hidden = hidden_node
        self.layers = layers
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_hidden_z = nn.Linear(n_hidden, n_hidden)
        self.linear_hidden_n = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)
        self.linear_input_z = nn.Linear(n_embed, n_hidden)
        self.linear_input_n = nn.Linear(n_embed, n_hidden)
        
    def forward (self, input_words):                    # => (batch size, sent len)
        
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)   #  (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)  # batch-node
        
        for i in range(input_words.size(1)):           #for i in seq_length

            ir=self.linear_input_r(embedded_words[i])
            hr=self.linear_hidden_r(hidden)
            r= ir.add(hr)
            rt = self.sigmoid(r)
            
            iz=self.linear_input_z(embedded_words[i])
            hz=self.linear_hidden_z(hidden)
            z= iz.add(hz)
            zt = self.sigmoid(z)
            
            
            iN=self.linear_input_n(embedded_words[i])
            hN=self.linear_hidden_n(hidden)*rt
            N= iN.add(hN)
            Nt = self.tanh(N)
            
            hidden = (1-zt)*Nt + zt*hidden
        
        out = self.fc(hidden)
        
        sig = self.sigmoid(out)
        return sig, hidden

class NetworkLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.embedding.weight=torch.nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        
        self.lstm = nn.LSTM(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
        
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        
        
    def forward (self, input_words):                       # => batch size, sent len
        embedded_words = self.embedding(input_words)    # => (batch_size, seq_length, n_embed)
        lstm_out, hidden = self.lstm(embedded_words)         # =>  (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, hidden

class NetworkLSTM_reimplement(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
        
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        self.hidden_node = hidden_node
        self.layers = layers
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        
    def forward (self, input_words):                    # => (batch size, sent len)
        
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)   #  (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)  # batch-node
        
        c = torch.zeros(input_words.size(0), self.hidden_node).to(device)
        
        for i in range(input_words.size(1)):           #for i in seq_length

            ir=self.linear_input_r(embedded_words[i])
            hr=self.linear_hidden_r(hidden)
            r= ir.add(hr)
            rt = self.sigmoid(r)
            
            iff=self.linear_input_f(embedded_words[i])
            hff=self.linear_hidden_f(hidden)
            ff= iff.add(hff)
            fft = self.sigmoid(ff)
            
            ig=self.linear_input_g(embedded_words[i])
            hg=self.linear_hidden_g(hidden)
            g= ig.add(hg)
            gt = self.tanh(g)
            
            io=self.linear_input_o(embedded_words[i])
            ho=self.linear_hidden_o(hidden)
            o= io.add(ho)
            ot = self.sigmoid(o)
            
            c = fft*c + rt*gt
            hidden = ot*self.tanh(c)
        
        out = self.fc(hidden)
        
        sig = self.sigmoid(out)
        return sig, hidden