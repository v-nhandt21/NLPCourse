
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer
import functools

device = "cuda"
import matplotlib.pyplot as plt

def loss_visual(name, train_loss, valid_loss):
    plt.plot(train_loss, color='blue')
    plt.plot(valid_loss, color='red')

    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,50,500))

    plt.savefig(name + '.png')

def NormData(filetext, filetag, fileout):
    with open(fileout, "w+", encoding="utf-8") as fw:
        fw.write("text\ttag\n")
        with open(filetext, "r",encoding="ISO-8859-1") as f:
            texts = f.read().splitlines()
            with open(filetag, "r",encoding="ISO-8859-1") as f1:
                tags = f1.read().splitlines()
                for text,tag in zip(texts,tags):
                    fw.write( text+ "\t"+tag+"\n")

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def LoadData():

    # TEXT = Field(init_token='<bos>', eos_token='<eos>', sequential=True)
    # TAGS = Field(init_token='<bos>', eos_token='<eos>', sequential=True, unk_token=None)

    TEXT = Field(sequential=True)
    TAGS = Field(sequential=True, unk_token=None)

    fields = [('text', TEXT), ('tag', TAGS)]
    train, valid, test = TabularDataset.splits(path="/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/", train='train.tsv', validation='dev.tsv',
                                            test='test.tsv', format='TSV', fields=fields, skip_header=True)

    
    
    TAGS.build_vocab(train)
    embed_len = 300
    TEXT.build_vocab(train, vectors="glove.6B.300d")
    
    
    
    device = "cuda"
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=1, device=device, train=False, shuffle=False, sort=False)

    return train_iter, valid_iter, test_iter, TAGS, TEXT, fields

if __name__ == '__main__':
    filetext = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/train.text"
    filetag = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/train.label"
    fileout = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/train.tsv"
    NormData(filetext,filetag,fileout)

    filetext = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/dev.text"
    filetag = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/dev.label"
    fileout = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/dev.tsv"
    NormData(filetext,filetag,fileout)

    filetext = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/test.text"
    filetag = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/test.label"
    fileout = "/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/test.tsv"
    NormData(filetext,filetag,fileout)
    
                