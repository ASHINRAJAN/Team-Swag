import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Streamlit UI
st.title('Iris Flower Classifier')

# Input for new sample
sepal_length = st.number_input('Enter Sepal Length:', min_value=0.0, step=0.1)
sepal_width = st.number_input('Enter Sepal Width:', min_value=0.0, step=0.1)
petal_length = st.number_input('Enter Petal Length:', min_value=0.0, step=0.1)
petal_width = st.number_input('Enter Petal Width:', min_value=0.0, step=0.1)

# Predict class of new sample
sample = [[sepal_length, sepal_width, petal_length, petal_width]]
predicted_class = clf.predict(sample)[0]

# Display prediction
st.write('The predicted class for the given sample is:', iris.target_names[predicted_class])
