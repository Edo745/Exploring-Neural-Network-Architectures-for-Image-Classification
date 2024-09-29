# Neural Network Architectures: Implementation and Comparison

This repository contains the implementation and comparison of three neural network architectures: a simple Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and a ResNet architecture. The aim is to explore and analyze their performance on a standard dataset, providing insights into their strengths and weaknesses for different types of tasks, particularly image classification.

A key objective of this project is to understand how skip connections work in deeper architectures like ResNet, and to assess how they help mitigate the vanishing gradient problem, which often affects very deep networks.

## Table of Contents
- [Introduction](#introduction)
- [Architectures](#architectures)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

Neural networks are widely used in machine learning for tasks like image classification, object detection, and more. This repository demonstrates the implementation of three key architectures:
1. **MLP (Multi-Layer Perceptron):** A simple feedforward neural network.
2. **CNN (Convolutional Neural Network):** Designed for image data, with layers that capture spatial hierarchies.
3. **ResNet (Residual Network):** A deeper architecture that uses skip connections to address the vanishing gradient problem.

The goal is to compare the performance of these architectures on the same dataset and analyze their trade-offs in terms of complexity, accuracy, and training time.

## Architectures

1. **MLP**  
   A basic feedforward neural network with fully connected layers. This is typically used for non-sequential data but can also be applied to flattened image data.

2. **CNN**  
   This network uses convolutional layers to automatically learn spatial features from the input images, followed by fully connected layers for classification.

3. **ResNet**  
   ResNet introduces residual connections (skip connections) to allow gradients to flow more easily through deeper networks, enabling better performance for more complex tasks.

## Dataset

The dataset used for training and evaluation is [insert dataset name, e.g., MNIST or CIFAR-10]. It contains [describe dataset contents], and it's commonly used for benchmarking deep learning models.

## Installation

To run the code in this repository, ensure you have the following prerequisites installed:

1. Python 3.x
2. PyTorch  
3. Torchvision
4. tqdm
5. WandB

### Install required libraries:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script for each model:

- For MLP:

  ```bash
  python main.py --model mlp --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log
  ```

- For CNN:

  ```bash
  python main.py --model cnn --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log
  ```

- For ResNet:

  ```bash
  python main.py --model resnet18 --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log
  ```

### Weight & Biases Integration
You can track experiments and compare results using Weights & Biases using --log argument.
Then, the training logs, metrics, and gradients will be automatically uploaded to WandB.

## Results
### Training and Validation Curves
The following plots show the training and validation loss and accuracy curves for the three models (MLP, CNN, ResNet):

- Training Loss & Accuracy:
![Training Loss](Lab1\results\train_loss.png)
![Training Accuracy](Lab1\results\train_acc.png)

- Validation Loss & Accuracy:
![Validation Loss](Lab1\results\val_loss.png)
![Validation Accuracy](Lab1\results\val_acc.png)

Each graph compares the three models (MLP, CNN, ResNet) to show their performance during training and validation. From these figures, we can observe that ResNet converges faster and reaches higher accuracy due to the effectiveness of skip connections in deeper networks.

### Gradients Analysis
One of the main goals of this project was to understand how skip connections in ResNet alleviate the vanishing gradient problem. The following gradient plots, taken from Weights & Biases, illustrate how gradients propagate through the layers of ResNet compared to the other architectures:

Gradients of ResNet (Layer: layer2.1.conv2.weight):

This plot shows that gradients in ResNet, thanks to the skip connections, maintain a higher magnitude across layers, preventing them from vanishing in deeper layers. This helps ResNet learn more effectively compared to MLP and CNN in very deep architectures.
### Performance Metrics:

- **MLP:**  
- **CNN:**  
- **ResNet:**  
 
 
