from model import BERT
from utils import LoadData
from utils import load_checkpoint
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
device = "cuda"
count = 0
def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    sumS = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for (labels, text), _ in test_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text, labels)

                output = torch.round(output.squeeze()).detach().cpu().numpy().astype(int)

                ground = labels.detach().cpu().numpy().astype(int)

                count = count + np.sum(output == ground)
                sumS = sumS + len(ground)
    
    print("Accuracy: " + str(count/sumS))

    # print('Classification Report:')
    # print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    

if __name__ == '__main__':
    train_iter, valid_iter, test_iter = LoadData()

    n_hidden = 512
    n_output = 1
    n_hidden_decode = 512
    v_attention_dimensionality = 512
    best_model = BERT(768, n_hidden, n_hidden_decode, n_output, v_attention_dimensionality).to(device)

    load_checkpoint("/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/Assignment4_BERTScratch/checkpoint" + '/model.pt', best_model)

    evaluate(best_model, test_iter)
