import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Define Node class for Decision Tree
class Node:
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute  # Attribute used for splitting
        self.value = value  # Value of the attribute
        self.result = result  # Class label if node is a leaf node
        self.children = {}  # Dictionary to store child nodes

# ID3 Algorithm
def id3(data, target, attributes):
    root = Node()
    if len(set(target)) == 1:
        root.result = target[0]  # If all target values are the same, return leaf node
        return root
    if len(attributes) == 0:
        root.result = max(set(target), key=target.count)  # Return the majority class label
        return root
    best_attribute = choose_attribute(data, target, attributes)
    root.attribute = best_attribute
    attributes.remove(best_attribute)
    for value in np.unique(data[best_attribute]):
        child = Node(attribute=best_attribute, value=value)
        root.children[value] = child
        subset_data = data[data[best_attribute] == value]
        subset_target = target[data[best_attribute] == value]
        if len(subset_data) == 0:
            child.result = max(set(target), key=target.count)  # If subset is empty, return majority class label
        else:
            child.children = id3(subset_data, subset_target, attributes.copy())
    return root

# Function to choose the best attribute for splitting
def choose_attribute(data, target, attributes):
    information_gain = []
    for attribute in attributes:
        information_gain.append(calc_information_gain(data, target, attribute))
    best_attribute_index = np.argmax(information_gain)
    return attributes[best_attribute_index]

# Function to calculate information gain
def calc_information_gain(data, target, attribute):
    total_entropy = entropy(target)
    attribute_values = np.unique(data[attribute])
    weighted_entropy = 0
    for value in attribute_values:
        subset_target = target[data[attribute] == value]
        weighted_entropy += (len(subset_target) / len(target)) * entropy(subset_target)
    return total_entropy - weighted_entropy

# Function to calculate entropy
def entropy(target):
    value_counts = pd.Series(target).value_counts()  # Count occurrences of each unique class label
    probabilities = value_counts / len(target)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

# Streamlit UI
st.title('ID3 Decision Tree Classifier')

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Select target column
target_column = st.selectbox("Select the target column", options=X.columns)

# Remove target column from list of attributes
attributes = list(X.columns)
attributes.remove(target_column)

# Build decision tree
root_node = id3(X, y, attributes)

# Display decision tree
st.write('Decision Tree:')
st.write(root_node)

# Input fields for user input
st.write('Enter values for the attributes to make a prediction:')
input_values = {}
for attribute in X.columns:
    if attribute != target_column:
        input_values[attribute] = st.number_input(f"Enter value for {attribute}")

# Predict function
def predict(root_node, input_values):
    current_node = root_node
    while current_node.children:
        attribute = current_node.attribute
        value = input_values[attribute]
        if value in current_node.children:
            current_node = current_node.children[value]
        else:
            return "Unable to make prediction"
    return current_node.result

# Make prediction
prediction = predict(root_node, input_values)

# Display prediction
st.write('Prediction:', iris.target_names[prediction])
