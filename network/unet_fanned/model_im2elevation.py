import torch

from network.unet_fanned.models import resnet, modules, net, densenet, senet


def define_model(is_resnet, is_densenet, is_senet):
    model = None

    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = modules.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = modules.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = modules.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    return model


if __name__ == '__main__':
    im2ele = define_model(is_resnet=True, is_senet=False, is_densenet=False)

    x = torch.randn(8, 3, 512, 512)

    out = im2ele(x)
    print(out.shape)