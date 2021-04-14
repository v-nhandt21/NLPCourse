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

class AttentionLSTM_Dot(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
        self.hidden_node = hidden_node
        self.layers = layers
        
        
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        
        self.fc_out_attent = nn.Linear(n_hidden*2, n_output)
        self.fc_out_lstm = nn.Linear(n_hidden, n_hidden)
        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
        
    def forward (self, input_words):                                                        # => (batch size, sent len)
        
        hidden_stack = []
        batch = input_words.shape[0]
        seq_lenght= input_words.size(1)
        
        embedded_words = self.embedding(input_words)                                        # => (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)              # batch-node
        
        c = torch.zeros(input_words.size(0), self.hidden_node).to(device)
        
        for i in range(seq_lenght):                                                         #for i in seq_length

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
            
            hidden_stack.append(hidden)
        
        
        final_hidden = self.fc_out_lstm(hidden)                                                      #[batch, hidden-node]
        
        outputs = torch.stack(hidden_stack).permute(1,0,2)                                           #[batch, seq-len, hidden-node]
        
        attention_score = torch.bmm(outputs,final_hidden.view(batch,self.hidden_node,1)).squeeze(2)  # [batch, seq-len, hidden-node] * [batch, hidden-node,1] => [batch, seq-len]
        
        attention_distribution = self.softmax(attention_score)
        
        attention_output = torch.bmm(
            outputs.permute(0,2,1),                                                                  #[batch, hidden-node, seq-len]
            attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
        ).squeeze(2)                                                                                 #[batch, hidden-node]
        
        
        out_atten = torch.cat([final_hidden, attention_output],dim=1)                                #[batch, hidden-node*2]  
        out = self.fc_out_attent(out_atten)                                                          #[batch, hidden-node]
        
        sig = self.sigmoid(out)
        return sig, hidden



class AttentionLSTM_Multiplicative(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, hidden_node_decode, n_output, layers):
        super().__init__()
        
        n_hidden = hidden_node
        n_hidden_decode = int(hidden_node_decode)
        self.hidden_node = hidden_node
        self.hidden_node_decode = int(hidden_node_decode)
        self.layers = layers
        
        
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        self.W_attention = nn.Parameter(torch.zeros([n_hidden_decode, n_hidden], device=device))
        self.b_attention = nn.Parameter(torch.zeros([n_hidden], device=device))
        
        
        self.fc_out_attent = nn.Linear(n_hidden + n_hidden_decode, n_output)
        self.fc_out_lstm = nn.Linear(n_hidden, n_hidden_decode)
        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
        
    def forward (self, input_words):                                                        # => (batch size, sent len)
        
        hidden_stack = []
        batch = input_words.shape[0]
        seq_lenght= input_words.size(1)
        
        embedded_words = self.embedding(input_words)                                        # => (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)              # batch-node
        
        c = torch.zeros(input_words.size(0), self.hidden_node).to(device)
        
        for i in range(seq_lenght):                                                         #for i in seq_length

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
            
            hidden_stack.append(hidden)
        
        
        final_hidden = self.fc_out_lstm(hidden)                                                      #[batch, hidden-node] => [batch, hidden-node-decode]
        
        outputs = torch.stack(hidden_stack).permute(1,2,0)                                           #[batch, hidden-node, seq-len]
        
        
        
        attention_score = torch.mm(final_hidden,self.W_attention) + self.b_attention                 # [batch,hidden-node-decode]*[hidden-node-decode, hidden-node]  # => [batch, hidden-node]

        attention_score = torch.bmm(attention_score.view(batch,1, self.hidden_node),outputs).squeeze(1)# batch, 1, hidden-node] *[batch, hidden-node, seq-len] => [batch, seq-len]
        
        
        attention_distribution = self.softmax(attention_score)                                       # [batch, seq-len]
        
        attention_output = torch.bmm(
            outputs,                                                                                 #[batch, hidden-node, seq-len]
            attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
        ).squeeze(2)                                                                                 #[batch, hidden-node]
        
        
        out_atten = torch.cat([final_hidden, attention_output],dim=1)                                #[batch, hidden-node*2]  
        out = self.fc_out_attent(out_atten)                                                          #[batch, hidden-node]
        
        sig = self.sigmoid(out)
        return sig, hidden



class AttentionLSTM_Additive(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, hidden_node_decode, n_output, layers, v_attention_dimensionality):
        super().__init__()
        
        n_hidden = hidden_node
        n_hidden_decode = int(hidden_node_decode)
        self.hidden_node = hidden_node
        self.hidden_node_decode = int(hidden_node_decode)
        self.layers = layers
        
        
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        # For Additive
        self.linear_additive_h = nn.Linear(n_hidden, v_attention_dimensionality)
        self.linear_additive_s = nn.Linear(n_hidden_decode, v_attention_dimensionality)
        
        self.linear_vdim = nn.Linear(v_attention_dimensionality,1)
        
        #self.W_Vdim = nn.Parameter(torch.zeros([v_attention_dimensionality], device=device))
        
        
        self.fc_out_attent = nn.Linear(n_hidden + n_hidden_decode, n_output)
        self.fc_out_lstm = nn.Linear(n_hidden, n_hidden_decode)
        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
        
    def forward (self, input_words):                                                        # => (batch size, sent len)
        
        hidden_stack = []
        batch = input_words.shape[0]
        seq_lenght= input_words.size(1)
        
        embedded_words = self.embedding(input_words)                                        # => (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)              # batch-node
        
        c = torch.zeros(input_words.size(0), self.hidden_node).to(device)
        
        for i in range(seq_lenght):                                                         #for i in seq_length

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
            
            hidden_stack.append(hidden)
        
        
        final_hidden = self.fc_out_lstm(hidden)                                                      #[batch, hidden-node] => [batch, hidden-node-decode]
        
        outputs = torch.stack(hidden_stack).permute(1,0,2)                                           #[batch, seq-len, hidden-node]
        
        WH = self.linear_additive_h(outputs)                                                         # => [batch, seq-len, v_dim]

        final_hidden_seq = final_hidden.repeat(1,seq_lenght)                                         #[batch, hidden-node-decode] => [batch, seq-len, hidden-node-decode]
        final_hidden_seq = final_hidden_seq.view(-1,seq_lenght, self.hidden_node_decode)             # ? Right way ? 
        
        
        WS = self.linear_additive_s(final_hidden_seq)                                                # => [batch, seq-len, v_dim]
        
        attention_score = self.tanh(WH+WS)  # => [batch, seq-len, v_dim]
        
        #print(self.W_Vdim.view(v_attention_dimensionality,1).shape)
        #attention_score = torch.mm(attention_score,self.W_Vdim.view(v_attention_dimensionality,1))  #[batch, seq-len]

        attention_score = self.linear_vdim(attention_score).squeeze(2)                               # ? Right way ? [batch, seq-len]
        
        attention_distribution = self.softmax(attention_score)                                       # [batch, seq-len]
        
        attention_output = torch.bmm(
            outputs.permute(0,2,1),                                                                                 #[batch, hidden-node, seq-len]
            attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
        ).squeeze(2)                                                                                 #[batch, hidden-node]
        
        
        out_atten = torch.cat([final_hidden, attention_output],dim=1)                                #[batch, hidden-node+ hidden-node-decoder]  
        out = self.fc_out_attent(out_atten)                                                          #[batch, hidden-node]
        
        sig = self.sigmoid(out)
        return sig, hidden



class AttentionBiLSTM_Additive(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, hidden_node_decode, n_output, layers, v_attention_dimensionality):
        super().__init__()
        
        n_hidden = hidden_node
        n_hidden_decode = int(hidden_node_decode)
        self.hidden_node = hidden_node
        self.hidden_node_decode = int(hidden_node_decode)
        self.layers = layers
        
        # LSTM1
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        # LSTM2
        self.linear_hidden_r2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r2 = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f2 = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g2 = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o2 = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o2 = nn.Linear(n_embed, n_hidden)
        
        
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        # For Additive
        self.linear_additive_h2 = nn.Linear(n_hidden*2, v_attention_dimensionality)
        self.linear_additive_s = nn.Linear(n_hidden_decode, v_attention_dimensionality)
        
        self.linear_vdim = nn.Linear(v_attention_dimensionality,1)
        
        self.fc_out_attent = nn.Linear(n_hidden*2 + n_hidden_decode, n_output)
        self.fc_out_attentA = nn.Linear(n_hidden*2 + n_hidden_decode, int((n_hidden*2 + n_hidden_decode)/2))
        self.fc_out_attentB = nn.Linear(int((n_hidden*2 + n_hidden_decode)/2), n_output)
        self.fc_out_lstm = nn.Linear(n_hidden, n_hidden_decode)
        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.3)
        
    def forward (self, input_words):                                                        # => (batch size, sent len)
        
        hidden_stack = []
        hidden_stack2 = []
        batch = input_words.shape[0]
        seq_lenght= input_words.size(1)
        
        embedded_words = self.embedding(input_words)                                        # => (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)              # batch-node
        
        c = torch.zeros(input_words.size(0), self.hidden_node).to(device)
        c2 = torch.zeros(input_words.size(0), self.hidden_node).to(device)
        
        for i in range(seq_lenght):                                                         #for i in seq_length

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
            
            hidden_stack.append(hidden)
            
        for i in range(seq_lenght-1, -1, -1):                                                         #for i in seq_length

            ir2=self.linear_input_r2(embedded_words[i])
            hr2=self.linear_hidden_r2(hidden)
            r2= ir2.add(hr2)
            rt2 = self.sigmoid(r2)
            
            iff2=self.linear_input_f2(embedded_words[i])
            hff2=self.linear_hidden_f2(hidden)
            ff2= iff2.add(hff2)
            fft2 = self.sigmoid(ff2)
            
            ig2=self.linear_input_g2(embedded_words[i])
            hg2=self.linear_hidden_g2(hidden)
            g2= ig2.add(hg2)
            gt2 = self.tanh(g2)
            
            io2=self.linear_input_o2(embedded_words[i])
            ho2=self.linear_hidden_o2(hidden)
            o2= io2.add(ho2)
            ot2 = self.sigmoid(o2)
            
            c2 = fft2*c2 + rt2*gt2
            hidden2 = ot2*self.tanh(c2)
            
            hidden_stack2.insert(0,hidden2)
        
        
        final_hidden = self.fc_out_lstm(hidden)                                                      #[batch, hidden-node] => [batch, hidden-node-decode]
        final_hidden = self.dropout(final_hidden)
        
        
        outputs1 = torch.stack(hidden_stack).permute(1,0,2)                                           #[batch, seq-len, hidden-node]
        outputs2 = torch.stack(hidden_stack2).permute(1,0,2)                                           #[batch, seq-len, hidden-node]

        outputs = torch.cat((outputs1,outputs2),2)                                                   #[batch, seq-len, hidden-node*2]
        WH = self.linear_additive_h2(outputs)                                                         # => [batch, seq-len, v_dim]
        WH = self.dropout(WH)
        
        final_hidden_seq = final_hidden.repeat(1,seq_lenght)                                         #[batch, hidden-node-decode] => [batch, seq-len, hidden-node-decode]
        final_hidden_seq = final_hidden_seq.view(-1,seq_lenght, self.hidden_node_decode)             # ? Right way ? 
        
        
        WS = self.linear_additive_s(final_hidden_seq)                                                # => [batch, seq-len, v_dim]
        WS = self.dropout(WS)
        
        attention_score = self.tanh(WH+WS)  # => [batch, seq-len, v_dim]

        attention_score = self.linear_vdim(attention_score).squeeze(2)                               # ? Right way ? [batch, seq-len]
        attention_score = self.dropout(attention_score)
        
        attention_distribution = self.softmax(attention_score)                                       # [batch, seq-len]
        
        attention_output = torch.bmm(
            outputs.permute(0,2,1),                                                                                 #[batch, hidden-node, seq-len]
            attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
        ).squeeze(2)                                                                                 #[batch, hidden-node]
        
        out_atten = torch.cat([final_hidden, attention_output],dim=1)                                #[batch, hidden-node*2+hidden-node-decoder]  
        
        #out = self.fc_out_attent(out_atten)                                                          #[batch, hidden-node]
        
#         print(out_atten.shape)
        out = self.fc_out_attentA(out_atten)
        out = self.dropout(out)
        out = self.fc_out_attentB(out)
        out = self.dropout(out)
        
        sig = self.sigmoid(out)
        return sig, hidden