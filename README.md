# **Alzheimer's Disease Classification using CNN (Convolutional Neural Network)**

## **Table of Contents**
- Overview
- Dataset
- Model Architecture
- Training & evaluation
- Requirements
- Results
  
## **Overview**
This project aims to classify Alzheimer's disease stages using a custom Convolutional Neural Network (CNN) built with PyTorch. 
The model categorizes brain MRI images into four classes:
- Very Mild
- Mild
- Moderate
- Non Mild

## **Dataset**
The dataset consists of MRI images structured into four labeled folders corresponding to the disease stages. 
The images are preprocessed and transformed with resizing, normalization, and data augmentation techniques to improve model performance. The dataset used is located
[Dataset here](https://github.com/MbungaiMichael/Alzheimer-s-Classification-using-CNN/tree/main/dataset)

## **Model Architecture**
The custom CNN consists of multiple convolutional layers, batch normalization, max pooling, and dropout for regularization. 
It is trained using the CrossEntropyLoss function with an Adam optimizer.

## **Training & Evaluation**
The model is trained on a dataset split into training and validation sets. 
The training loop updates weights using backpropagation, while the validation set is used to monitor performance and avoid overfitting.
The project is located [Project here](https://github.com/MbungaiMichael/Alzheimer-s-Classification-using-CNN/blob/main/Alzheimer_classification.ipynb)
## **Requirements**
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

## **Results**
The model achieves high accuracy in distinguishing between different stages of Alzheimer's, helping in early detection and diagnosis.
Validation accuracy shows a consistent improvement across epochs, indicating effective learning. 
The confusion matrix demonstrates that the model correctly classifies most images, with minor misclassifications in borderline cases. 
Further improvements, such as fine-tuning hyperparameters and adding more training data, could enhance performance even further.



