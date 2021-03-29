from model import BERT
from utils import LoadData, save_checkpoint, save_metrics
import torch
import torch.optim as optim
import torch.nn as nn

train_iter, valid_iter, test_iter = LoadData()
device = "cuda"

def train(model,optimizer,criterion = nn.BCELoss(),train_loader = train_iter,valid_loader = valid_iter,
        num_epochs = 4,
        eval_every = len(train_iter) // 2,
        file_path = "/home/ubuntu/NLPCourse/Assignment/Sentiment_analysis/Assignment4_BERTScratch",
        best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):

        # count = 0
        # if epoch < 3:
        #     for param in net.children():
        #         count +=1
        #         if count < 4: #freezing first 3 layers
        #             param.requires_grad = False
        # else:
        #     for param in net.children():
        #         param.requires_grad = True


        for (labels, text), _ in train_loader:

            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            output = model(text, labels)

            loss = criterion(output.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (labels, text), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)           
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)  
                        text = text.to(device)
                        output = model(text, labels)
                        
                        loss = criterion(output.squeeze(), labels.float())
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

if __name__ == '__main__':
    n_hidden = 512
    n_output = 1
    n_hidden_decode = 512
    v_attention_dimensionality = 512
    model = BERT(768, n_hidden, n_hidden_decode, n_output, v_attention_dimensionality).to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train(model=model, optimizer=optimizer)
