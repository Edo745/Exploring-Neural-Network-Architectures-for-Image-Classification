# Exploring Neural Network Architectures for Image Classification
This repository contains the implementation and comparison of different neural network architectures: a simple Multi-Layer Perceptron (MLP), a Convolutional Neural Network, a deeper Convolutional Neural Network and two ResNet architectures. The aim is to explore and analyze their performance on CIFAR10, providing insights into their strengths and weaknesses for image classification and trying to replicate the results highlighted from the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385). The following points will be explored:

- **MLP vs CNN**: Which architecture performs better in image recognition?
- **CNN vs deeper CNN**: Is it true that the deeper a model is, the better its performance?
- **Residual Networks**: The benefits of residual connections
- **CNN vs ResNet**: How residual connections can improve the accuracy and efficiency of the model
- **GradCAM**: Which are the image regions the model focuses on? Can we know why the model made that prediction? 
  
## Dataset
The dataset used for training and evaluation is CIFAR-10.
<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/174468b7-0010-4d91-9d8c-8d9bf9f73847">
</p>

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

## Experiments
### MLP vs CNN: Which architecture performs better in image recognition?  
<p align="center">
  <img src="https://github.com/user-attachments/assets/fd8d3d1e-29a4-4723-b1a6-ebb57ec1b936" width="400" alt="Train Accuracy" title="Train Accuracy"/>
  <img src="https://github.com/user-attachments/assets/38657932-3cd4-4243-a247-9dc6c7f67322" width="400" alt="Test Accuracy" title="Test Accuracy"/>
</p> 

As we can observe the CNN outperforms the MLP. The latter has the disadvantage of having too many parameters since it consists of fully connected layers where each node is connected to all others. Additionally, it takes flattened vectors as inputs, which disregards spatial information. In contrast, in a CNN, the layers are sparsely/partially connected, and thanks to convolutional operations, it can preserve spatial relationships in the data, capturing important features. This ability allows CNNs to handle image data more efficiently and achieve better performance in tasks like image recognition.

### CNN19 vs CNN30: Is it true that the deeper a model is, the better its performance? 
<p align="center">
  <img src="https://github.com/user-attachments/assets/64c83487-c922-47c2-bf46-6d9293dddd15" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/a101566d-6a57-4efb-89b1-c4a672753664" width="400" alt="Test Loss" title="Train Loss"/> 
</p> 

Observing the first epochs, the shallower CNN converges more quickly. As the epochs progress, CNN30 manages to achieve better accuracy, but this is also due to the phenomenon of overfitting. In general, as demonstrated in the paper (https://arxiv.org/abs/1512.03385), very deep architectures suffer from the problem of gradient degradation. 

### CNN vs ResNet: How residual connections can improve the accuracy and efficiency of the model 
<p align="center">
  <img src="https://github.com/user-attachments/assets/7526c49f-9d23-447e-921e-38360e908ca6" width="400" alt="Train Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/cd17ac09-5507-4e34-a85c-74d19bcde505" width="400" alt="Test Loss" title="Train Loss"/> 
</p> 

<p align="center">
  <img src="https://github.com/user-attachments/assets/d1f6e3fa-c007-4d05-a4af-b83f2f6d2d77" width="400" alt="Test Loss" title="Train Loss"/> 
  <img src="https://github.com/user-attachments/assets/3868584f-a62f-4ad1-a87e-01fbebd4cdd2" width="400" alt="Train Loss" title="Train Loss"/> 
</p> 

To address the problem of gradient vanishing, the authors of the paper (https://arxiv.org/abs/1512.03385) propose the use of residual connections, providing a 'shortcut' for gradients to flow backwards directly from later layers to earlier layers. As we can observe from the figures, thanks to residual connections, there is faster convergence and better performance.

This highlights how residual connections mitigate gradient vanishing and improve training efficiency and accuracy in deep networks.

## GradCAM: Which are the image regions the model focuses on?
Let's implement Grad-CAM following the paper [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391).

The idea is to analyze the gradients and the activations to gain insights into why the model made predictions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8300c781-45bb-4606-8f4e-27766ac40fa9" width="900"/> 
</p> 

1. The gradient of the feature map with respect to a target class is computed, obtaining k derivatives ((those colored in the figure).
2. The global average pooling of each derivative is computed, obtaining k important weights.
3. A linear combination is performed between the weights and the original feature maps (+ in the figure).
4. A ReLU is applied because we are interested in the positive influence of each pixel.

Run the gradcam.py script:
```bash
!python gradcam.py --model cnn19 --label "horse" --img-size 32 --target-layer "model.conv_block3[9]"
!python gradcam.py --model cnn19 --label "frog" --img-size 32 --target-layer "model.conv_block3[9]"

!python gradcam.py --model cnn30 --label "horse" --img-size 32 --target-layer "model.conv_block3[9]" 
!python gradcam.py --model cnn30 --label "frog" --img-size 32 --target-layer "model.conv_block3[9]"

!python gradcam.py --model resnet18 --label "horse" --img-size 32 --target-layer "model.layer4[-1]" 
!python gradcam.py --model resnet18 --label "frog" --img-size 32 --target-layer "model.layer4[-1]"
```
### cnn19 vs cnn30 vs resnet18 - horse
<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/61055a96-f270-4f9d-a0c4-9a56043f02e8" alt="GradCAM_cnn19_horse_model conv_block3" title="GradCAM CNN19 Horse" />
  <img width="600" src="https://github.com/user-attachments/assets/602521e6-574f-48c4-94ae-6da584b2792f" alt="GradCAM_cnn30_horse_model conv_block3" title="GradCAM CNN30 Horse" />
  <img width="600" src="https://github.com/user-attachments/assets/fda4f04f-4832-4552-b809-ca5cbf4db784" alt="GradCAM_resnet18_horse_model layer4" title="GradCAM ResNet18 Horse" />
</p>

### cnn19 vs cnn30 vs resnet18 - frog
<p align="center">
  <img width="600"  src="https://github.com/user-attachments/assets/c92d03e6-0803-4811-9d0f-284aa4287e2a" alt="GradCAM_cnn19_frog_model conv_block3" title="GradCAM CNN19 Frog" />
  <img width="600" src="https://github.com/user-attachments/assets/d84d4b69-9e43-436f-9eae-ce9fbb18088d" alt="GradCAM_cnn30_frog_model conv_block3" title="GradCAM CNN30 Frog" />
  <img width="600" src="https://github.com/user-attachments/assets/e1780401-52fd-4f4e-9543-91ecb15cc3c9" alt="GradCAM_resnet18_frog_model layer4" title="GradCAM ResNet18 Frog" />
</p>

# Take home message
- The convolutional neural networks outperform MLPs for the task of image recognition. Convolutional layers naturally retain spatial
information which is lost in fully-connected layers.

- As the depth of a neural network increases beyond a certain point, its performance on the training and test sets starts to degrade due to the vanishing of the gradients.
  
- Residual connections addresses the degradation problem providing a "shortcut" for gradients to flow backwards directly from later layers to earlier layers. This helps mitigate the vanishing gradient problem and enables the training of significantly deeper networks.
<p align="center">
  <img width="300" src="https://github.com/user-attachments/assets/45554c19-b234-468a-8142-e2a5e8f40437">
</p>
  
- ResNets generally have smaller magnitudes of responses compared to plain networks. This means that residual functions are closer to zero than non-residual functions, making them easier to optimize. In other words, the model has to learn just small corrections.
  
- Residual connections are great but be carefull with too deep architectures, they can lead to overfitting!

- Analyzing the gradients and the activations we can gain insights into why the model made predictions (GradCAM).
