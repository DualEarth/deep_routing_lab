# Simple CNN Example

## Overview

***
This file serves as a format for developing more advanced neural networks. It contains python files defining the CNN model and its configuration, outlining data specifications, trains the model, and visualizes data into images. The goal of this neural network to be able to be find patterns in inputted pixel images that can be built upon later. 

## Components

***

#### cnn_model.py

A customizable CNN for processing and combining two image inputs, ideal for regression or comparison tasks.

##### Key Features

* Dual input paths with shared convolutional layers.
* Configurable parameters: image_size, hidden_size, kernel_size.
* Fully connected layers for feature aggregation and prediction.
* Optional verbose mode for debugging.

#### config.py

Defines key parameters for the CNN and training process.

#### data_generation.py

This script generates synthetic datasets for training and evaluation, used for tasks such as image prediction or comparison.

##### Key Features

* Data Generation:
	* X1: Generated as variations of the sine function, modified by random multipliers, additions, and rotations.
	* X2: Generated similarly, but using the cosine function with different random transformations.
	* y: The target data is the element-wise product of X1 and X2.
* Data Splitting:
	* The synthetic data is split into training and evaluation datasets based on num_images_train, with corresponding DataLoader objects for each set.

#### evaluate_model.py

This code evaluates the trained PyTorch model on a dataset and visualizes the results.

##### Key Features

* The evaluate_model function evaluates a model's performance on a given dataset using Mean Squared Error (MSE) loss.
* Automatically detects whether a GPU (CUDA) is available and moves the model and data to the appropriate device (CPU or GPU).
* Uses nn.MSELoss() to compute the loss between the model's predictions and the true labels.
* Visualizes the first batch's inputs, outputs, and labels.
* Calculates and prints the average evaluation loss after processing the entire dataset.

#### train_model.py

The train_model function trains the neural network model using the synthetic dataset.

## Problems

***
* A fair  amount of the values (such as image_size) are hardcoded and would require scaling for larger inputs. 
* The neural network only contains 2 convolutional layers which may require scaling for more advanced tasks. 
* Inefficient nested loops for data generation especially for larger datasets and image sizes.

## Future Development

This example of basic image-based CNN serves to show a way that they can be trained to analyze patterns. Future work should be aimed at making the network more robust so that it can be adapted to a wider array of uses and applications. 
