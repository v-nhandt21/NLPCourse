#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"


# In[2]:


from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import sys
import os
import string, nltk
#nltk.download('stopwords')
nltk.data.path.append("/home/ubuntu/nltk_data")
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
#nltk.download('punkt')
#nltk.download('wordnet')
import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torchsummary import summary


# # Utils Functions

# In[3]:




Vocab = []
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english') + list(string.punctuation)) 

def Norm(text,wordnet_lemmatizer,stop_words):
    text = text.lower().replace("\s+"," ")
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            w = wordnet_lemmatizer.lemmatize(w, pos="v")
            filtered_sentence.append(w) 
    texts=" ".join(str(x) for x in filtered_sentence)
    return text

def pad_text(encoded_reviews, seq_length):
    
    reviews = []
    
    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0]*(seq_length-len(review)) + review)
        
    return np.array(reviews)

def LoadData(file, Vocab=Vocab):
    with open(file, "r",encoding="ISO-8859-1") as f:
        data_x = []
        data_y = []
        contents = f.read().splitlines()
        for line in contents:
            try:
                _,text,label = line.split("#")
            except:
                continue
            text = text.split(" ",1)[1]
            
            text = "".join([ch for ch in text if ch not in punctuation])
            text = Norm(text,wordnet_lemmatizer,stop_words)
            
            data_x.append(text)
            data_y.append(label)
            Vocab = Vocab + text.split(" ")
        return data_x, data_y, Vocab


# In[ ]:





# # Data preprocessing

# In[4]:


train_x, train_y, Vocab = LoadData("../data/train.txt",Vocab)
dev_x, dev_y, Vocab = LoadData("../data/dev.txt",Vocab)
test_x, test_y, Vocab = LoadData("../data/test.txt",Vocab)


# In[5]:


print(train_x[:100])


# In[6]:


word_counts = Counter(Vocab)
word_list = sorted(word_counts, key = word_counts.get, reverse = True)
vocab_to_int = {word:idx+1 for idx, word in enumerate(word_list)}
int_to_vocab = {idx:word for word, idx in vocab_to_int.items()}


# In[7]:


encoded_train = [[vocab_to_int[word] for word in review.split(" ")] for review in train_x]
train_x = pad_text(encoded_train, seq_length = 200)
train_y = np.array([1 if label == "pos" else 0 for label in train_y])

encoded_dev = [[vocab_to_int[word] for word in review.split(" ")] for review in dev_x]
dev_x = pad_text(encoded_dev, seq_length = 200)
dev_y = np.array([1 if label == "pos" else 0 for label in dev_y])

encoded_test = [[vocab_to_int[word] for word in review.split(" ")] for review in test_x]
test_x = pad_text(encoded_test, seq_length = 200)
test_y = np.array([1 if label == "pos" else 0 for label in test_y])


# In[8]:



train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)


# In[9]:


torch.randn((50,300), requires_grad=True, dtype=torch.float)


# # Models

# In[10]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NetworkLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        self.lstm = nn.LSTM(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
        
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):                       # => batch size, sent len
        embedded_words = self.embedding(input_words)    # => (batch_size, seq_length, n_embed)
        lstm_out, hidden = self.lstm(embedded_words)         # =>  (batch_size, seq_length, n_hidden)
        
        out = self.fc(lstm_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, hidden

class NetworkGRU_(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        self.Wi = nn.Parameter(torch.randn( (n_embed,hidden_node), requires_grad=True, dtype=torch.float))
        self.Wh = nn.Parameter(torch.randn( (hidden_node,hidden_node) , requires_grad=True, dtype=torch.float))
        
        self.hidden_node = hidden_node
        self.layers = layers
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward (self, input_words):                    # => (batch size, sent len)
        
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)   #  (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)  # batch-node
        
        for i in range(input_words.size(1)):           #for i in seq_length

            A=embedded_words[i].matmul(self.Wi)
            B=hidden.matmul(   self.Wh)
            C = A.add(B)
            hidden = self.sigmoid(C)
        
        out = self.fc(hidden)
        
        sig = self.sigmoid(out)
        return sig, hidden
    
class NetworkRNN_(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        self.Wi = nn.Parameter(torch.randn( (n_embed,hidden_node), requires_grad=True, dtype=torch.float))
        self.Wh = nn.Parameter(torch.randn( (hidden_node,hidden_node) , requires_grad=True, dtype=torch.float))
        
        self.hidden_node = hidden_node
        self.layers = layers
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
#         self.rnn = nn.RNN(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
#         self.rnn_cell = nn.RNNCell(n_embed, hidden_node)
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward (self, input_words):                    # => (batch size, sent len)
        
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        embedded_words = embedded_words.permute(1,0,2)   #  (seq_length,batch_size,  n_embed)
        hidden = torch.zeros(input_words.size(0), self.hidden_node).to(device)  # batch-node
        
        for i in range(input_words.size(1)):           #for i in seq_length

            A=embedded_words[i].matmul(self.Wi)
            B=hidden.matmul(   self.Wh)
            C = A.add(B)
            hidden = self.sigmoid(C)
        
        out = self.fc(hidden)
        
        sig = self.sigmoid(out)
        return sig, hidden
    
class NetworkGRU(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        self.gru = nn.GRU(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
        
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        gru_out, hidden = self.gru(embedded_words)         # (batch_size, seq_length, n_hidden)
        
        out = self.fc(gru_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, hidden
    
    
    
    
    
    
class NetworkRNN(nn.Module):
    
    def __init__(self, n_vocab, n_embed, hidden_node, n_output, layers):
        super().__init__()
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        
        self.rnn = nn.RNN(n_embed, hidden_node, layers, batch_first = True, bidirectional=False)
        
        self.fc = nn.Linear(n_hidden, n_output)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        gru_out, hidden = self.rnn(embedded_words)         # (batch_size, seq_length, n_hidden)
        
        out = self.fc(gru_out[:, -1, :])
        
        sig = self.sigmoid(out)
        return sig, hidden


# # Training

# In[11]:



n_vocab = len(vocab_to_int)
n_embed = 300
n_hidden = 256 #512
n_output = 1   # 1 ("positive") or 0 ("negative")
layers = 1

net = NetworkRNN_(n_vocab, n_embed, n_hidden, n_output, layers).cuda()

criterion = nn.BCELoss()
criterion = criterion.cuda()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

print(net)

# inp = torch.zeros((1,200), dtype=torch.long) # [length, batch_size]
# print(summary(net,(300) ))


# In[ ]:



print_every = 100
step = 0
n_epochs = 8 #4
clip = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
for epoch in range(n_epochs):
    
    for inputs, labels in train_loader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        net.zero_grad()
        output, h = net(inputs)
        try:
            loss = criterion(output.squeeze(), labels.float())
        except:
            output[output < 0.0] = 0.0
            output[output > 1.0] = 1.0
            loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()
        
        if (step % print_every) == 0:            
            net.eval()
            valid_losses = []
            
            for v_inputs, v_labels in valid_loader:
                v_inputs, v_labels = inputs.to(device), labels.to(device)

                
                v_output, v_h = net(v_inputs)
                v_loss = criterion(v_output.squeeze(), v_labels.float())
                valid_losses.append(v_loss.item())

            print("Epoch: {}/{}".format((epoch+1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            net.train()
            
torch.save(net.state_dict(), "LSTM.pt")


# # Inference Sample

# In[ ]:


def inference(net, review, seq_length = 200):
    device = "cuda" #"cuda" if torch.cuda.is_available() else "cpu"
    
    text = review.lower()
    text = "".join([ch for ch in text if ch not in punctuation])
    words = text
    
    encoded_words = [vocab_to_int[word] for word in words.split(" ")]
    padded_words = pad_text([encoded_words], seq_length)
    padded_words = torch.from_numpy(padded_words).to(device)

    
    net.eval().to(device)
    output, h = net(padded_words )#, h)
    pred = torch.round(output.squeeze())  
    return pred


# In[ ]:


inference(net, "I am sad") 


# # Predict test

# In[ ]:


net.eval().to(device)
count = 0
for v_inputs, v_labels in test_loader:
    v_inputs, v_labels = inputs.to(device), labels.to(device)

    v_output, v_h = net(v_inputs)

    output = torch.round(v_output.squeeze()).detach().cpu().numpy().astype(int)
    #print(len(output))
    ground = labels.detach().cpu().numpy().astype(int)
    #print(len(ground))
    count = count + np.sum(output == ground)
print(count/len(test_x))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[161]:


def predict(checkpoint, test_file, Vocab):

    #     n_vocab = len(vocab_to_int)
#     n_embed = 400
#     n_hidden = 512
#     n_output = 1   # 1 ("positive") or 0 ("negative")
#     layers = 1

#     net = NetworkRNN(n_vocab, n_embed, n_hidden, n_output, layers).cuda()
    
#     test_x, test_y, Vocab = LoadData("../data/test.txt",Vocab)

#     net.load_state_dict(torch.load("LSTM.pt"))
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net.eval().to(device)
    count = 0
    for v_inputs, v_labels in test_loader:
        v_inputs, v_labels = inputs.to(device), labels.to(device)

        v_output, v_h = net(v_inputs)

        output = torch.round(v_output.squeeze()).detach().cpu().numpy().astype(int)
        #print(len(output))
        ground = labels.detach().cpu().numpy().astype(int)
        #print(len(ground))
        count = count + np.sum(output == ground)
    print(count/len(test_x))


# In[162]:


predict("LSTM.pt","../data/test.txt",Vocab)


# In[ ]:





# In[ ]:




