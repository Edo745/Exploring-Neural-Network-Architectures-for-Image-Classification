# The deeper neural networks the better?
This repository contains the implementation and comparison of different neural network architectures: a simple Multi-Layer Perceptron (MLP), a Convolutional Neural Network, a deeper Convolutional Neural Network and a ResNet architecture. The aim is to explore and analyze their performance on a standard dataset, providing insights into their strengths and weaknesses for image classification. Di seguito i punti che verranno approfonditi:

- **MLP vs CNN**
- **CNN vs deeper CNN**: è vero che tanto più è profondo un modello tanto più aumentano le performance?
- **Residual Connections**: come risolvere il problema dei vanishing gradients.

## Table of Contents
- [Introduction](#introduction)
- [Architectures](#architectures)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Architectures
Di seguito le architetture implementate:
1. **mlp**  
   A basic feedforward neural network with fully connected layers.
   
3. **cnn19**  

4. **cnn30**  

5. **rescnn30**  
   

## Dataset

The dataset used for training and evaluation is CIFAR-10.

### Install required libraries:
To run the code in this repository, ensure you have the following prerequisites installed:

1. Python 3.x
2. PyTorch  
3. Torchvision
4. tqdm
5. wandb
   
```bash
pip install -r requirements.txt
```

## Usage

Run the training script for each model:

- For mlp:

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
![train_loss](https://github.com/user-attachments/assets/167840b1-f7f6-4f5b-8063-6eeb51fb58f8)
![train_acc](https://github.com/user-attachments/assets/6450fad8-7e8a-4008-9a8a-157c3aeb9849)

- Validation Loss & Accuracy:
![val_loss](https://github.com/user-attachments/assets/92bd3ba1-50db-49f7-88cb-2c63e02ac1e6)
![val_acc](https://github.com/user-attachments/assets/3664cbe2-a46a-4813-9fd1-f1a3eb6d26dc)


Each graph compares the three models (MLP, CNN, ResNet) to show their performance during training and validation. From these figures, we can observe that ResNet converges faster and reaches higher accuracy due to the effectiveness of skip connections in deeper networks.

## Conclusions
- Convolutional neural networks outperform MLPs for the task of image recognition.
- In very deep neural networks, as we backpropagate the gradients from the output layer to the input layer, these gradients can become extremely small (vanish). This makes it difficult for the network to learn and can lead to poor performance.
- Residual connections, discussed in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385), provide a "shortcut" for gradients to flow backwards directly from later layers to earlier layers. Mathematically, if $x$ is the input and $F(x)$ is the output of a block of layers, a residual connection would compute: $y = F(x) + x$.
- Residual connections are great but be carefull with too deep architectures, they can lead to overfitting!
