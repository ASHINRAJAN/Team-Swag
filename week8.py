import streamlit as st
import numpy as np

# Define the features (sepal length, sepal width, petal length, petal width)
features = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [5.4, 3.7, 1.5, 0.2],
    [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1],
    [4.3, 3.0, 1.1, 0.1],
    [5.8, 4.0, 1.2, 0.2]
])

# Define the target labels (0: setosa, 1: versicolor, 2: virginica)
targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Define the target names
target_names = np.array(['setosa'])

# Combine features and targets into a dictionary similar to the Iris dataset
iris_dataset = {
    'data': features,
    'target': targets,
    'target_names': target_names,
    'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
}

# Streamlit code to display the dataset
st.title("Sample of the Iris dataset")

for i in range(5):
    st.write(f"Sample {i+1}:")
    st.write("  Features:", iris_dataset['data'][i])
    st.write("  Target:", iris_dataset['target'][i], "(", iris_dataset['target_names'][iris_dataset['target'][i]], ")")
    st.write("")
