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

<p align="center">
  <img width="720" src="https://github.com/user-attachments/assets/bb0e9bfd-b0dc-497f-9c2b-bc33a852d0ea">
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
  <img src="https://github.com/user-attachments/assets/fd8d3d1e-29a4-4723-b1a6-ebb57ec1b936" width="400" alt="Train Accuracy" title="Train Accuracy"/>
  <img src="https://github.com/user-attachments/assets/38657932-3cd4-4243-a247-9dc6c7f67322" width="400" alt="Test Accuracy" title="Test Accuracy"/>
</p> 



### CNN19 vs CNN30  
<p align="center">
  <img src="https://github.com/user-attachments/assets/64c83487-c922-47c2-bf46-6d9293dddd15" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/a101566d-6a57-4efb-89b1-c4a672753664" width="400" alt="Test Loss" title="Train Loss"/> 
</p> 


### CNN30 vs ResidualCNN30  
<p align="center">
  <img src="https://github.com/user-attachments/assets/7526c49f-9d23-447e-921e-38360e908ca6" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/cd17ac09-5507-4e34-a85c-74d19bcde505" width="400" alt="Test Loss" title="Train Loss"/> 
</p> 

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
