# Estimation of Obesity Levels Based on Eating Habits and Physical Condition Using Artificial Neural Network

## Overview

This project develops an Artificial Neural Network (ANN) to predict obesity levels based on individuals' eating habits and physical conditions. Utilizing the Multilayer Perceptron algorithm, this model aims to provide a more accurate and comprehensive understanding of obesity determinants, offering a practical tool for personal health assessment and research purposes.

## Prerequisites

Before you can run this project, you'll need to have the following installed on your system:

- Python 3.7 or higher
- Pip (Python package installer)

Additionally, this project depends on several Python libraries, including:

- Numpy
- Pandas
- Matplotlib

## Installation

To set up this project on your local machine, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/VedatYalcinkaya/Estimation-of-obesity-with-Neural-Network.git
```
Install the required Python packages:
```bash
pip install numpy pandas matplotlib
```
## Background

Obesity is a globally prevalent condition, with its incidence rising dramatically due to changes in lifestyle and dietary habits. Early detection and awareness of obesity levels can lead to more effective interventions and lifestyle adjustments. This project leverages machine learning to analyze and predict obesity levels, contributing valuable insights into the obesity epidemic.

## Dataset

The dataset comprises responses from individuals in Peru, Colombia, and Mexico, totaling 2111 samples. It includes 17 attributes related to eating habits, physical condition, and demographic information. The target attribute classifies individuals into one of seven obesity levels, from Underweight to Obesity Type III.

Dataset Source: [UCI Machine Learning Repository - Estimation of Obesity Levels Based on Eating Habits and Physical Condition Dataset](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+)

## Methodology

### Multilayer Perceptron (MLP)

The project employs a Multilayer Perceptron, a class of feedforward artificial neural network, to handle the complexity of predicting obesity levels. The ANN architecture includes an input layer corresponding to the dataset features, several hidden layers to process the data, and an output layer that predicts the obesity level.

![image](https://github.com/VedatYalcinkaya/Estimation-of-obesity-with-Neural-Network/assets/87366287/8828db71-ac1b-43df-931f-6f58ba7569a1)

### Data Preprocessing

- **Categorical to Numerical**: Transforms categorical data into numerical values for model processing.
- **Normalization**: Applies normalization to feature values to improve model performance.
- **Randomization**: Ensures data is randomly shuffled for training and testing to prevent bias.

### Training and Testing

The model is trained with a subset of the dataset and tested against the remaining data to evaluate its accuracy and effectiveness. The ANN's performance is measured based on its ability to accurately classify the obesity level of unseen data.

## Results

The ANN achieved an optimal accuracy rate of up to 94.4% in predicting the correct obesity level among seven categories. This high level of accuracy demonstrates the model's potential as a reliable tool for obesity level estimation based on non-invasive measures.

![Accuracy](https://github.com/VedatYalcinkaya/Estimation-of-obesity-with-Neural-Network/assets/87366287/c3dcfcbd-353a-4b3c-ba34-38673c9d58c9)
![image](https://github.com/VedatYalcinkaya/Estimation-of-obesity-with-Neural-Network/assets/87366287/34149675-2e0a-497e-b45d-f40b4cb23baf)


## Technologies

- **Python**: The primary programming language used for developing the ANN model.
- **Libraries**: NumPy for numerical operations, Pandas for data manipulation, and Matplotlib for visualization.

## How to Contribute

Contributors are welcome to improve the project, add features, or resolve issues. Please follow the contribution guidelines provided in this repository.
