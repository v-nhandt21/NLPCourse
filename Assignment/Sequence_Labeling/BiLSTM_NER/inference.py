from model import BERT_Seqence
from utils import LoadData
from utils import load_checkpoint
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
device = "cuda"
count = 0
def evaluate(model, test_loader, tagvocab):
    model.eval()

    

    with open("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/predict.txt", "w+", encoding="utf-8") as fw:
        with torch.no_grad():
            for (text,tags), _ in test_loader:
                predictions = model(text)
                
                top_predictions = predictions.argmax(-1)
        
                predicted_tags = [tagvocab.vocab.itos[t.item()] for t in top_predictions]
                
                predicted_tags = predicted_tags[1:]

                fw.write(" ".join(predicted_tags) + "\n")
                print(predicted_tags)
    
def check_infer():
    with open("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/test.label", "r", encoding="utf-8") as fr:
        text_infer = fr.read().splitlines()
    with open("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/predict.txt", "r", encoding="utf-8") as fr1:
        text_infer1 = fr1.read().splitlines()
    for t, t1 in zip(text_infer, text_infer1):
        assert len(t.split(" ")) == len(t1.split(" "))

if __name__ == '__main__':
    train_iter, valid_iter, test_iter , tagvocab= LoadData()

    n_output = len(tagvocab.vocab)
    best_model = BERT_Seqence(768, n_output).to(device)

    load_checkpoint("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/BERT_NER" + '/model.pt', best_model)

    evaluate(best_model, test_iter, tagvocab)

    check_infer()

 # Accuracy: 0.8668575518969219