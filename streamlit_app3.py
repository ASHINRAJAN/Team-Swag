import streamlit as st
import numpy as np
import pandas as pd

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
    classes, counts = np.unique(target, return_counts=True)
    probabilities = counts / len(target)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

# Streamlit UI
st.title('ID3 Decision Tree Classifier')

# Synthetic dataset generation
@st.cache
def generate_data():
    np.random.seed(0)
    data = pd.DataFrame({
        'Outlook': np.random.choice(['Sunny', 'Overcast', 'Rainy'], size=100),
        'Temperature': np.random.choice(['Hot', 'Mild', 'Cool'], size=100),
        'Humidity': np.random.choice(['High', 'Normal'], size=100),
        'Windy': np.random.choice(['Weak', 'Strong'], size=100),
        'PlayTennis': np.random.choice(['Yes', 'No'], size=100)
    })
    return data

data = generate_data()
st.write('Generated Dataset:')
st.write(data)

# Select target column
target_column = st.selectbox("Select the target column", options=data.columns)

# Remove target column from list of attributes
attributes = list(data.columns)
attributes.remove(target_column)

# Build decision tree
root_node = id3(data, data[target_column], attributes)

# Display decision tree
st.write('Decision Tree:')
st.write(root_node)

# Input fields for user input
st.write('Enter values for the attributes to make a prediction:')
input_values = {}
for attribute in data.columns:
    if attribute != target_column:
        input_values[attribute] = st.selectbox(f"Select value for {attribute}", options=data[attribute].unique())

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
st.write('Prediction:', prediction)
