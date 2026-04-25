import torchvision.models as models
import timm
import torch

device = "cpu"

def save_models(model_name):
    dim=224

    if model_name == 'rn50':
        model = models.__dict__['resnet50'](pretrained=True).to(device)

    elif model_name == 'instagram_resnext101_32x8d':
        dim=32
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)

    elif model_name == 'bit_m_rn50':
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model = model.to(device)

    elif model_name == 'rn18':
        model = models.__dict__['resnet18'](pretrained=True).to(device)
    
    model.eval()
    traced = torch.jit.trace(model, (torch.rand(4, 3, dim, dim),))
    torch.jit.save(traced, "pretrained_models/{}.pt".format(model_name))

    # load model
    model = torch.jit.load("pretrained_models/{}.pt".format(model_name))
    inputs = torch.rand(4, 3, dim, dim)
    s = model(inputs)
    print(s)
    
if __name__ == '__main__':
    for name in ['rn18', 'rn50', 'bit_m_rn50']:
        save_models(name)

