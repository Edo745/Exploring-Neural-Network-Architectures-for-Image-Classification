def get_model(model_name):
    if model_name == "mlp":
        from models.mlp import MLP
        model = MLP()
    elif model_name == "cnn":
        from models.cnn import CNN
        model = CNN()
    elif model_name == "resnet18":
        from models.resnet18 import ResNet18
        model = ResNet18()
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model


    