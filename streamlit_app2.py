import streamlit as st
import numpy as np

# Function to normalize input
def normalize_input(X):
    return X / np.amax(X, axis=0)

# Function for sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function for forward propagation
def forward_propagation(X, wh, bh, wout, bout):
    h_ip = np.dot(X, wh) + bh
    h_act = sigmoid(h_ip)
    o_ip = np.dot(h_act, wout) + bout
    output = sigmoid(o_ip)
    return output

# Load weights
wh = np.array([[0.13, 0.68, 0.80], [0.87, 0.63, 0.25]])  # Example weights
bh = np.array([[0.4, 0.5, 0.6]])  # Example biases
wout = np.array([[0.2], [0.3], [0.7]])  # Example weights
bout = np.array([[0.1]])  # Example bias

# Streamlit UI
st.title('Neural Network Predictor')

st.write('Enter values to predict the output:')

# Input fields
x1 = st.number_input('Input 1', value=2.0)
x2 = st.number_input('Input 2', value=9.0)

X = np.array([[x1, x2]])

# Normalize input
X = normalize_input(X)

# Prediction
output = forward_propagation(X, wh, bh, wout, bout)

st.write('Predicted Output:', output)
