
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer

device = "cuda"

def LoadData():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('label', label_field), ('text', text_field)]
    train, valid, test = TabularDataset.splits(path="/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data", train='train.tsv', validation='dev.tsv',
                                            test='test.tsv', format='TSV', fields=fields, skip_header=True)

    device = "cuda"
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)


    return train_iter, valid_iter, test_iter


def NormData(filein, fileout):
    with open(fileout, "w+", encoding="utf-8") as fw:
        fw.write("label\ttext\n")
        with open(filein, "r",encoding="ISO-8859-1") as f:
            data_x = []
            data_y = []
            contents = f.read().splitlines()
            for line in contents:
                try:
                    _,text,label = line.split("#")
                except:
                    print(line)
                    print("\n\n\n")
                    continue
                text = text.split(" ",1)[1]
                if label == "pos":
                    label = "1"
                else:
                    label = "0"
                fw.write( label+ "\t"+text+"\n")


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



if __name__ == '__main__':
    filein = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data/train.txt"
    fileout = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data/train.tsv"

    filein1 = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data/dev.txt"
    fileout1 = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data/dev.tsv"

    filein2 = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data/test.txt"
    fileout2 = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/data/test.tsv"

    NormData(filein,fileout)
    NormData(filein1,fileout1)
    NormData(filein2,fileout2)
                