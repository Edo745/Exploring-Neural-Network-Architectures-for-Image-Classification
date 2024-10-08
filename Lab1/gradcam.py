import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision import datasets
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import argparse
import os
from torch.utils.data import DataLoader

class GradCAM:
  def __init__(self, model, target_layer):
      self.model = model
      self.target_layer = target_layer
      self.gradients = None
      self.activations = None

      self.hook_activation()
      self.hook_gradient()

  def hook_activation(self):
    def forward_hook(module, input, output):
          self.activations = output
    self.target_layer.register_forward_hook(forward_hook)

  def hook_gradient(self):
      def backward_hook(module, grad_input, grad_output):
          self.gradients = grad_output[0]

      self.target_layer.register_full_backward_hook(backward_hook)

  def generate_cam(self, input_image, class_index):
      # Forward pass
      self.model.eval()
      output = self.model(input_image)

      # Backward pass
      self.model.zero_grad()
      output[0, class_index].backward()

      gradients = self.gradients.cpu().data.numpy()[0]
      activations = self.activations.cpu().data.numpy()[0]

      weights = np.mean(gradients, axis=(1, 2))

      cam = np.zeros(activations.shape[1:], dtype=np.float32)

      for i, w in enumerate(weights):
          cam += w * activations[i]

      # ReLU e normalizzazione
      cam = np.maximum(cam, 0)
      cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
      cam = cam - np.min(cam)
      cam = cam / np.max(cam)
      return cam, output

  def overlay_cam(self, img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img - np.min(img)
    img = img / np.max(img)

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return cam_img

def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def main():
    parser = argparse.ArgumentParser(description='GradCAM')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', help='Path to the checkpoints')
    parser.add_argument('--label', type=str, default=None, help='Class label to visualize')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained model')
    parser.add_argument('--target_layer', type=str, default="model.layer4[1].conv1", help='Number of convolutional layer')
    args = parser.parse_args()
    
    if not os.path.exists('results'):
      os.makedirs('results')

    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    label_to_class_idx = {label: idx for idx, label in enumerate(labels)}

    if args.label not in label_to_class_idx:
        raise ValueError(f"Label '{args.label}' not found. Choose from {labels}.")
    
    class_idx = label_to_class_idx[args.label]
    
    if args.model == 'resnet18':
      if args.pretrained:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
      else:
        model = models.resnet18()
        #model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet50':
        if args.pretrained:
          model = models.resnet50(pretrained=True)
          model.fc = nn.Linear(model.fc.in_features, 10)
        else:
          model = models.resnet50()
          model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'cnn19':
        from models.cnn19 import CNN19 
        model = CNN19()
    elif args.model == 'cnn30':
        from models.cnn30 import CNN30 
        model = CNN30()
    elif args.model == 'cnn30residual':
        from models.cnn30residual import CNN30Residual 
        model = CNN30Residual()
    if not args.pretrained:
      checkpoint = torch.load(f"{args.checkpoints_dir}/{args.model}_best.pth", map_location=torch.device('cpu'), weights_only=True)
      model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    input_image = None

    for img, lbl in dataloader:
        if lbl.item() == class_idx:
          input_image = img.squeeze(0)
          break
    

    inverse_normalize(input_image)
    plt.imshow(input_image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Label: {labels[class_idx]}')
    plt.savefig(f"results/original_{args.label}.png")
    plt.show()

    target_layer = eval(args.target_layer)
    grad_cam = GradCAM(model, target_layer)
    
    input_image = input_image.unsqueeze(0)  
    cam, output = grad_cam.generate_cam(input_image, class_idx)

    overlayed_image = grad_cam.overlay_cam(input_image, cam)

    plt.imshow(overlayed_image)
    plt.axis('off')
    plt.title(f'GradCAM {labels[class_idx]}\nPredicted: {labels[output.argmax().item()]}')
    plt.savefig(f"results/CamGRAD_{args.target_layer}_{args.label}.png")
    plt.show()

if __name__ == '__main__':
    main()
