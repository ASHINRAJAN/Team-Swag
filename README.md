# Neural Network with Backpropagation

This code implements a simple neural network with backpropagation using Python and NumPy.

## Overview

- `neural_network.py`: Python script containing the neural network implementation.
- `README.md`: This file, providing an overview of the code and instructions for usage.

## Requirements

- Python 3.x
- NumPy

## Usage

1. Clone the repository:
git clone https://github.com/your_username/your_repository.git

css
Copy code

2. Navigate to the directory:
cd your_repository

markdown
Copy code

3. Run the script:
python neural_network.py

markdown
Copy code

## Description

The neural network implemented in `neural_network.py` consists of an input layer, a hidden layer, and an output layer. The activation function used is the sigmoid function.

- `sigmoid(x)`: Sigmoid activation function.
- `sigmoid_grad(x)`: Derivative of the sigmoid function.

The network is trained using backpropagation for a specified number of epochs (`epoch`). It takes input data `X` and corresponding output labels `y`, normalizes the input, and then updates the weights and biases iteratively using the backpropagation algorithm.

After training, the script prints the normalized input, actual output, and predicted output.

## Parameters

- `epoch`: Number of training epochs.
- `eta`: Learning rate.
- `input_neurons`: Number of neurons in the input layer.
- `hidden_neurons`: Number of neurons in the hidden layer.
- `output_neurons`: Number of neurons in the output layer.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.# Team-Swag
