import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertGenerationEncoder, BertModel
import seaborn as sns


device="cuda"

# Additive Attention LSTM using Pretrain BERT
class BERT_LSTM(nn.Module):

    def __init__(self, n_embed, hidden_node, hidden_node_decode, n_output, v_attention_dimensionality):
        super(BERT, self).__init__()

        self.encoder = BertModel.from_pretrained("bert-base-uncased")


        n_hidden = hidden_node
        n_hidden_decode = int(hidden_node_decode)
        self.hidden_node = hidden_node
        self.hidden_node_decode = int(hidden_node_decode) 
        
        self.linear_hidden_r = nn.Linear(n_hidden, n_hidden)
        self.linear_input_r = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_f = nn.Linear(n_hidden, n_hidden)
        self.linear_input_f = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_g = nn.Linear(n_hidden, n_hidden)
        self.linear_input_g = nn.Linear(n_embed, n_hidden)

        self.linear_hidden_o = nn.Linear(n_hidden, n_hidden)
        self.linear_input_o = nn.Linear(n_embed, n_hidden)
        
        # For Additive
        self.linear_additive_h = nn.Linear(n_hidden, v_attention_dimensionality)
        self.linear_additive_s = nn.Linear(n_hidden_decode, v_attention_dimensionality)
        
        self.linear_vdim = nn.Linear(v_attention_dimensionality,1)
        
        self.fc_out_attent = nn.Linear(n_hidden + n_hidden_decode, n_output)
        self.fc_out_lstm = nn.Linear(n_hidden, n_hidden_decode)
        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, label):
        text_fea = self.encoder(text).last_hidden_state                                    # (batch, seq, emb)

        hidden_stack = []
        batch = text_fea.shape[0]
        seq_lenght= text_fea.size(1)

        embedded_words = text_fea.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(text_fea.size(0), self.hidden_node).to(device)              # batch-node
        
        c = torch.zeros(text_fea.size(0), self.hidden_node).to(device)
        
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
        
        attention_score = self.tanh(WH+WS)                                                           # => [batch, seq-len, v_dim]

        attention_score = self.linear_vdim(attention_score).squeeze(2)
        
        attention_distribution = self.softmax(attention_score)                                       # [batch, seq-len]
        
        attention_output = torch.bmm(
            outputs.permute(0,2,1),                                                                                 #[batch, hidden-node, seq-len]
            attention_distribution.view(batch,seq_lenght,1)                                          #[batch, seq_len, 1]
        ).squeeze(2)                                                                                 #[batch, hidden-node]
        
        
        out_atten = torch.cat([final_hidden, attention_output],dim=1)                                #[batch, hidden-node+ hidden-node-decoder]  
        out = self.fc_out_attent(out_atten)                                                          #[batch, hidden-node]
        
        sig = self.sigmoid(out)
        return sig










import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertGenerationEncoder, BertModel
import seaborn as sns


device="cuda"

# Additive Attention LSTM using Pretrain BERT
class BERT_Seqence(nn.Module):

    def __init__(self, n_embed, n_output):
        super(BERT_Seqence, self).__init__()

        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(n_embed,  n_output)        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        text_fea = self.encoder(text).last_hidden_state                                    # (batch, seq, emb)

        hidden_stack = []
        batch = text_fea.shape[0]
        seq_lenght= text_fea.size(1)

        embedded_words = text_fea.permute(1,0,2)                                      # => (seq_length,batch_size,  n_embed)
        
                    
        #embedded = [sent len, batch size, emb dim]
        
        predictions = self.fc(self.dropout(embedded_words))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions






