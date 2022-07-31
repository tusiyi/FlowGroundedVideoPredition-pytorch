import torch
import torch.nn as nn
import torchvision.models as models


def get_features(vgg19, input_x):
    x = vgg19.features[0](input_x)
    relu_1 = vgg19.features[1](x)
    x = vgg19.features[2](relu_1)
    relu_2 = vgg19.features[3](x)
    x = vgg19.features[4](relu_2)
    x = vgg19.features[5](x)
    relu_3 = vgg19.features[6](x)
    x = vgg19.features[7](relu_3)
    relu_4 = vgg19.features[8](x)
    x = vgg19.features[9](relu_4)
    x = vgg19.features[10](x)
    relu_5 = vgg19.features[11](x)

    return [relu_1, relu_2, relu_3, relu_4, relu_5]


if __name__ == '__main__':

    model = models.vgg19(pretrained=True)
    x1 = torch.randn((1, 3, 512, 512))
    with torch.no_grad():
        features = get_features(model, x1)
    print(model)
