from model import BERT
from utils import LoadData
from utils import load_checkpoint
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

device = "cuda"

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, text), _ in test_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())

    accuracy = [ m-n for m,n in zip(y_pred,y_true) ].count(0) / len(y_true)
    print("Accuracy: " + str(accuracy))

    # print('Classification Report:')
    # print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    


train_iter, valid_iter, test_iter = LoadData()

best_model = BERT().to(device)

load_checkpoint("/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/Assignment4_BERTClassifier/checkpoint" + '/model.pt', best_model)

evaluate(best_model, test_iter)