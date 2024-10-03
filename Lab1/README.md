# Exploring Neural Network Architectures for Image Classification
This repository contains the implementation and comparison of different neural network architectures: a simple Multi-Layer Perceptron (MLP), a Convolutional Neural Network, a deeper Convolutional Neural Network and a ResNet architecture. The aim is to explore and analyze their performance on a standard dataset, providing insights into their strengths and weaknesses for image classification nad trying to replicate the results highlighted from the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385). The following points will be explored:

- **MLP vs CNN**: Which architecture performs better in image recognition?
- **CNN vs deeper CNN**: Is it true that the deeper a model is, the better its performance?
- **Residual Networks**: The benefits of residual connections
  
## Dataset
The dataset used for training and evaluation is CIFAR-10.

## Implemented architectures
<p align="center">
  <img width="150" src="https://github.com/user-attachments/assets/e24b34c6-0915-46f3-ae58-2be1e03d8869">
</p>

<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/2cfc3467-d180-4fab-91a4-20c39daee5e3">
</p>

<p align="center">
  <img width="715" src="https://github.com/user-attachments/assets/2ab00c18-46d8-4a1e-b0c6-89f1a898cb19">
</p>


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
### MLP vs CNN19  

<p align="center">
  <img src="https://github.com/user-attachments/assets/957e0444-36c3-4949-8da9-310d22937d1f" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/5a17b226-359e-4827-8a24-be20c45cf134" width="400" alt="Train Accuracy" title="Train Accuracy"/>
</p> 

<p align="center">
  <img src="https://github.com/user-attachments/assets/badc4ffe-9867-4cfb-a8ac-96032bac1632" width="400" alt="Validation Loss" title="Validation Loss"/> 
  <img src="https://github.com/user-attachments/assets/a59e2fc9-da16-4817-afd5-83a2c8b07e4d" width="400" alt="Validation Accuracy" title="Validation Accuracy"/>
</p> 

### CNN19 vs CNN30  
<p align="center">
  <img src="https://github.com/user-attachments/assets/c17ff6d6-5f78-49ee-832d-54f3ba37975c" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/13ce45db-dd02-418e-ac5b-3ff2346493eb" width="400" alt="Train Accuracy" title="Train Accuracy"/>
</p> 

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3ba1b6a-7be5-4033-8a7d-fbc31a4ef229" width="400" alt="Validation Loss" title="Validation Loss"/> 
  <img src="https://github.com/user-attachments/assets/2d28a100-8b2b-48a1-8995-0c28a5570d39" width="400" alt="Validation Accuracy" title="Validation Accuracy"/>
</p> 

### CNN30 vs ResidualCNN30  

### Layer responses analysis
The paper highlights that ResNets generally have smaller magnitudes of responses compared to plaid networks. Let's prove it!

## Can we make the model explanable?
Let's implement Grad-CAM following the paper [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391).
# Take home message
- The convolutional neural networks outperform MLPs for the task of image recognition. Convolutional layers naturally retain spatial
information which is lost in fully-connected layers.

- As the depth of a neural network increases beyond a certain point, its performance on the training and test sets starts to degrade due to the vanishing of the gradients.
  
- Residual connections addresses the degradation problem providing a "shortcut" for gradients to flow backwards directly from later layers to earlier layers. This helps mitigate the vanishing gradient problem and enables the training of significantly deeper networks.
<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/45554c19-b234-468a-8142-e2a5e8f40437">
</p>
  
- ResNets generally have smaller magnitudes of responses compared to plain networks. This means that residual functions are closer to zero than non-residual functions, making them easier to optimize. In other words, the model has to learn just small corrections.
  
- Residual connections are great but be carefull with too deep architectures, they can lead to overfitting!

- Analyzing the gradients and the activations we can gain insights into why the model made predictions.
