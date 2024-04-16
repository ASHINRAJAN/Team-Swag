import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define Decision Tree classifier
class DecisionTreeClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self.predict_sample(sample))
        return predictions

    def predict_sample(self, sample):
        # Simple decision tree based on petal width
        if sample[3] < 0.8:
            return 0
        elif sample[3] < 1.7:
            return 1
        else:
            return 2

# Load the Iris dataset
def load_iris():
    from sklearn import datasets
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

X, y, feature_names, target_names = load_iris()

# Streamlit UI
st.title('Iris Flower Classifier')

st.write('Enter the features of the flower to predict its class:')

# Input fields
sepal_length = st.number_input('Sepal Length', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, step=0.1)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Predict class of new sample
sample = [sepal_length, sepal_width, petal_length, petal_width]
predicted_class = target_names[clf.predict_sample(sample)]

# Display prediction
st.write('Predicted Class:', predicted_class)

# Decision tree visualization
st.write('Decision Tree Visualization:')
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Decision Tree Classification')
plt.show()
st.pyplot()
