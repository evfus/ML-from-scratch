import config
import dataset
import model
import torch

def generate():
    #model = torch.load(...)

    context = torch.zeros((1, 1), dtype = torch.long, device = config.device)
    output = model.generate(context, 2000)[0].tolist()
    output = decode(output)

    return output
