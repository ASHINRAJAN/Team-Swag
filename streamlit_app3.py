import streamlit as st
import pandas as pd
import numpy as np

# Function to classify instance using the decision tree
def classify(instance, tree, default=None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classify(instance, result)
        else:
            return result
    else:
        return default

# Load the tennis dataset
df_tennis = pd.read_csv('tennis2.csv')

# ID3 algorithm
def id3(df, target, attribute_name, default_class=None):
    cnt = Counter(x for x in df[target])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attribute_name):
        return default_class
    else:
        default_class = max(cnt.keys())
        gains = [info_gain(df, attr, target) for attr in attribute_name]
        index_max = gains.index(max(gains))
        best_attr = attribute_name[index_max]
        tree = {best_attr: {}}
        remaining_attr = [x for x in attribute_name if x != best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target, remaining_attr, default_class)
            tree[best_attr][attr_val] = subtree
        return tree

# Streamlit UI
st.title('Tennis Play Prediction')

# Input fields for user input
st.write('Enter values for the attributes to predict PlayTennis:')
outlook = st.selectbox('Outlook', df_tennis['Outlook'].unique())
temperature = st.selectbox('Temperature', df_tennis['Temperature'].unique())
humidity = st.selectbox('Humidity', df_tennis['Humidity'].unique())
windy = st.selectbox('Windy', df_tennis['Windy'].unique())

# Predict function
def predict(outlook, temperature, humidity, windy):
    instance = {'Outlook': outlook, 'Temperature': temperature, 'Humidity': humidity, 'Windy': windy}
    result = classify(instance, tree, 'No')  # Default to 'No' if prediction cannot be made
    return result

# Make prediction
prediction = predict(outlook, temperature, humidity, windy)

# Display prediction
st.write('Prediction:', prediction)
