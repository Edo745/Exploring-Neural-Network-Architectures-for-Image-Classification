# Exploring Neural Network Architectures for Image Classification
This repository contains the implementation and comparison of different neural network architectures: a simple Multi-Layer Perceptron (MLP), a Convolutional Neural Network, a deeper Convolutional Neural Network and a ResNet architecture. The aim is to explore and analyze their performance on a standard dataset, providing insights into their strengths and weaknesses for image classification nad trying to replicate the results highlighted from the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385). The following points will be explored:

- **MLP vs CNN**: Which architecture performs better in image recognition?
- **CNN vs deeper CNN**: Is it true that the deeper a model is, the better its performance?
- **Residual Networks**: The benefits of residual connections
  
## Dataset
The dataset used for training and evaluation is CIFAR-10.

## Implemented architectures
1. **mlp**  
   A basic feedforward neural network with fully connected layers.
   
3. **cnn19**  

4. **cnn30**  

5. **residual_cnn30**

7. **resnet18**
   
9. **resnet50**  

## Usage
Install the requirements:
```bash
pip install -r requirements.txt
```

Run the training script for each model:
  ```bash
  python main.py --model resnet18 --epochs 50 --batch-size 128 --lr 0.1 --num-workers 2 --log
  ```

## Weight & Biases Integration
You can track experiments and compare results using Weights & Biases using --log argument.
Then, the training logs, metrics, and gradients will be automatically uploaded to WandB.

## Results
### Training Loss & Accuracy
The following plots show the training and validation loss and accuracy curves for the three models (MLP, CNN, ResNet):
<p align="center">
  <img src="https://github.com/user-attachments/assets/167840b1-f7f6-4f5b-8063-6eeb51fb58f8" width="500" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/6450fad8-7e8a-4008-9a8a-157c3aeb9849" width="500" alt="Train Accuracy" title="Train Accuracy"/>
</p> 



### Validation Loss & Accuracy
<p align="center">
  <img src="https://github.com/user-attachments/assets/92bd3ba1-50db-49f7-88cb-2c63e02ac1e6" width="500" alt="Validation Loss" title="Validation Loss"/> 
  <img src="https://github.com/user-attachments/assets/3664cbe2-a46a-4813-9fd1-f1a3eb6d26dc" width="500" alt="Validation Accuracy" title="Validation Accuracy"/>
</p> 

Each graph compares the three models (MLP, CNN, ResNet) to show their performance during training and validation. From these figures, we can observe that ResNet converges faster and reaches higher accuracy due to the effectiveness of skip connections in deeper networks.

### Layer responses analysis
The paper highlights that ResNets generally have smaller magnitudes of responses compared to plaid networks. Let's prove it!

# What we have learned
- The convolutional neural networks outperform MLPs for the task of image recognition. The reason is that convolutional layers naturally retain spatial
information which is lost in fully-connected layers.

- As the depth of a neural network increases beyond a certain point, its performance on the training and test sets starts to degrade due to the vanishing of the gradients.
  
- Residual connections addresses the degradation problem providing a "shortcut" for gradients to flow backwards directly from later layers to earlier layers. This helps mitigate the vanishing gradient problem and enables the training of significantly deeper networks.
<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/45554c19-b234-468a-8142-e2a5e8f40437">
</p>
  
- The analysis of layer responses shows that ResNets generally have smaller magnitudes of responses compared to plain networks. This means that residual functions are closer to zero than non-residual functions, making them easier to optimize. In other words, the model has to learn just small corrections.
  
- Residual connections are great but be carefull with too deep architectures, they can lead to overfitting!
