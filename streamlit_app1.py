import numpy as np 
import pandas as pd
import streamlit as st

def learn(concepts, target): 
    specific_h = concepts[0].copy()  
    st.write("Initialization of specific_h:\n", specific_h)  
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]     
    st.write("Initialization of general_h:\n", general_h)  

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            st.write("If instance is Positive")
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    specific_h[x] = '?'                     
                    general_h[x][x] = '?'
                   
        if target[i] == "no":            
            st.write("If instance is Negative")
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        

        st.write("Step", i+1)
        st.write("specific_h:\n", specific_h)         
        st.write("general_h:\n", general_h)
        st.write("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        general_h.remove(['?', '?', '?', '?', '?', '?']) 
    return specific_h, general_h 

def main():
    st.title("Candidate Elimination Algorithm")
    
    uploaded_file = st.file_uploader("C:\Users\user\Documents\cand\trainingdata.csv", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        concepts = np.array(data.iloc[:, 0:-1])
        target = np.array(data.iloc[:, -1])
        
        if st.button("Run Algorithm"):
            with st.spinner('Running algorithm...'):
                s_final, g_final = learn(concepts, target)
                st.success("Algorithm completed successfully!")
                st.write("Final Specific_h:")
                st.write(s_final)
                st.write("Final General_h:")
                st.write(g_final)

if _name_ == "_main_":
    main()
     
