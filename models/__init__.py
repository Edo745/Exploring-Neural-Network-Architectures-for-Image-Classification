import torch.nn as nn
from models.mlp import MLP
from models.cnn19 import CNN19
from models.cnn30 import CNN30
from models.resnet18 import ResNet18
from models.cnn30residual import CNN30Residual
import torchvision
from torchvision.models import ResNet
import torch 

def get_model(model_name):
    num_classes = 10   

    if model_name == "mlp":
        model = MLP()
    elif model_name == "cnn19":
        model = CNN19()
    elif model_name == "cnn30":
        model = CNN30()   
    elif model_name == "cnn30residual":
        model = CNN30Residual()  
    elif model_name == "resnet18":
        model = ResNet18()
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model
