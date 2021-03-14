import torch

def np_to_torch(npdata):
    '''
    transform numpy array to torch tensor
    :param npdata: a numpy array
    :return: a torch tensor
    '''
    data = npdata.astype('float')
    data = torch.from_numpy(data)
    # torch_data = torch.tensor(data, dtype=torch.float32)
    torch_data = data.clone().detach().float()
    return torch_data
