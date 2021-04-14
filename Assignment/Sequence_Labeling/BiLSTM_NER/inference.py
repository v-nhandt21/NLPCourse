from utils import LoadData
from utils import load_checkpoint
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from torchcrf import CRF
from model import AttentionBiLSTM_Seqence
import torchtext

train_iter, valid_iter, test_iter, TAGS, TEXT, fields = LoadData()

device = "cuda"
count = 0
    
def check_infer():
    with open("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/test.label", "r", encoding="utf-8") as fr:
        text_infer = fr.read().splitlines()
    with open("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/predict.txt", "r", encoding="utf-8") as fr1:
        text_infer1 = fr1.read().splitlines()
    for t, t1 in zip(text_infer, text_infer1):
        print("ok")
        assert len(t.split(" ")) == len(t1.split(" "))

def evaluate(model):
        # This method applies the trained model to a list of sentences.
        
        # First, create a torchtext Dataset containing the sentences to tag.
        crf = CRF(len(TAGS.vocab)).to(device)

        model.eval()
        out = []
        with open("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/predict_bilstm.txt", "w+", encoding="utf-8") as fw:
            with torch.no_grad():
                for (text,tags), _ in test_iter:
                    output = model(text,tags)
                    top_predictions = crf.decode(output)

                    predicted_tags = [TAGS.vocab.itos[t] for t in top_predictions[0] ]
                
                    predicted_tags = predicted_tags[1:]

                    fw.write(" ".join(predicted_tags) + "\n")
                    print(predicted_tags)


if __name__ == '__main__':

    best_model = AttentionBiLSTM_Seqence(TEXT, TAGS, n_embed=300, n_hidden=128).to("cuda")

    load_checkpoint("/home/ubuntu/NLPCourse/Assignment/Sequence_Labeling/BiLSTM_NER" + '/model.pt', best_model)

    evaluate(best_model)

    check_infer()

 # Accuracy: 0.8668575518969219
