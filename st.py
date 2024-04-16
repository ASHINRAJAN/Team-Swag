import streamlit as st
import pandas as pd
from collections import Counter
import math

# Define functions for ID3 algorithm
def entropy_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instance = len(a_list) * 1.0
    probs = [x / num_instance for x in cnt.values()]
    return entropy(probs)

def entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])

def info_gain(df, split, target, trace=0):
    df_split = df.groupby(split)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target: [entropy_list, lambda x: len(x) / nobs]})
    df_agg_ent.columns = ['Entropy', 'PropObserved']
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent["PropObserved"])
    old_entropy = entropy_list(df[target])
    return old_entropy - new_entropy

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

# Load data
df_tennis = pd.read_csv('tennis2.csv')

# Streamlit UI
st.title('Tennis Play Predictor')

# Input fields for user input
input_values = {}
for attribute in df_tennis.columns:
    if attribute != 'PlayTennis':
        input_values[attribute] = st.selectbox(f"Select value for {attribute}", options=df_tennis[attribute].unique())

# Train decision tree
attribute_names = list(df_tennis.columns)
attribute_names.remove('PlayTennis')
tree = id3(df_tennis, 'PlayTennis', attribute_names)

# Predict outcome
predicted_outcome = classify(input_values, tree)

# Display prediction
st.write('Predicted Outcome:', predicted_outcome)
