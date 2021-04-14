import torch
device="cuda"
def save_checkpoint(save_path, model):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict()}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])