import timm
import torch.nn as nn
import torch

class clc_model(nn.Module):
    def __init__(self,  model_name, no_cls):
        super(clc_model, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=True)
        if model_name == "resnext50_32x4d" or model_name == "resnet18":
            self.model.fc = nn.Linear(self.model.fc.in_features, no_cls)
        else:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, no_cls)

    def forward(self, x):
        return self.model(x)
    
# # test code here
# model = clc_model("mobilenetv3_large_100", no_cls=3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)
# print(model)