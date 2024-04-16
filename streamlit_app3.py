import streamlit as st
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Define function to predict class of new sample
def predict_class(sample):
    prediction = clf.predict([sample])
    return iris.target_names[prediction[0]]

# Streamlit UI
st.title('Iris Flower Classifier')

st.write('Enter the features of the flower to predict its class:')

# Input fields
sepal_length = st.number_input('Sepal Length', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, step=0.1)

# Predict class of new sample
sample = [sepal_length, sepal_width, petal_length, petal_width]
predicted_class = predict_class(sample)

# Display prediction
st.write('Predicted Class:', predicted_class)

# Display decision tree visualization
st.write('Decision Tree Visualization:')
plt.figure(figsize=(10, 7))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
st.pyplot()
