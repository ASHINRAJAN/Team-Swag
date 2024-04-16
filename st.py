import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import math

st.title("Decision Tree Classifier")

@st.cache
def entropy_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instance = len(a_list)*1.0
    probs = [x/num_instance for x in cnt.values()]
    return entropy(probs)

def entropy(probs):
    return sum([-prob*math.log(prob,2) for prob in probs])

def info_gain(df,split,target,trace=0):
    df_split = df.groupby(split)
    nobs = len(df.index)*1.0
    df_agg_ent = df_split.agg({ target:[entropy_list, lambda x: len(x)/nobs] })
    df_agg_ent.columns = ['Entropy','PropObserved']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent["PropObserved"])
    old_entropy = entropy_list(df[target])
    return old_entropy - new_entropy

def id3(df,target,attribute_name,default_class = None):
    cnt = Counter(x for x in df[target])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or (not attribute_name):
        return default_class
    else:
        default_class = max(cnt.keys())
        gains = [info_gain(df,attr,target) for attr in attribute_name]
        index_max = gains.index(max(gains))
        best_attr = attribute_name[index_max]
        tree = { best_attr:{ } }
        remaining_attr = [x for x in attribute_name if x!=best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,target,remaining_attr,default_class)
            tree[best_attr][attr_val] = subtree
        return tree

def classify(instance,tree,default = None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result,dict):
            return classify(instance,result)
        else:
            return result
    else:
        return default

def main():
    st.sidebar.title("Decision Tree Classifier")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Training Data Set:")
        st.write(df)

        attribute_names = list(df.columns)
        attribute_names.remove('PlayTennis') # Remove the class attribute
        tree = id3(df, 'PlayTennis', attribute_names)
        st.subheader("Resultant Decision Tree:")
        st.write(tree)

        test_data = df.iloc[-4:] # just the last four rows
        test_data['predicted'] = test_data.apply(classify, axis=1, args=(tree,'Yes') )
        accuracy = sum(test_data['PlayTennis'] == test_data['predicted']) / len(test_data.index)
        st.subheader("Testing and Accuracy:")
        st.write("The Accuracy for the test data is: ", accuracy)

if __name__ == "__main__":
    main()
