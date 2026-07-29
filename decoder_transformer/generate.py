import config
import dataset
from model import DecoderTransformer
import torch

def generate():
    model = DecoderTransformer().to(config.device)
    
    checkpoint = torch.load(
            "outputs/checkpoints/last.pt",
            map_location = config.device
    )

    model.load_state_dict(checkpoint["model"])

    context = torch.zeros((1, 1), dtype = torch.long, device = config.device)
    output = model.generate(context, 2000)[0].tolist()
    output = dataset.decode(output)

    return output
