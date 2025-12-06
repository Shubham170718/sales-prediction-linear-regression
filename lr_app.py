import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

from pyexpat import features

model = pickle.load(open('linear_regression_model.pkl','rb'))

# let's create web app
st.title("Scikit-learn Linear Regression model")
tv=st.text_input('enter tv sales....')
radio=st.text_input('enter radio sales...')
newspaper=st.text_input('enter newspaper sales...')

if st.button("predict"):
    features=np.array([[tv,radio,newspaper]],dtype=np.float64)
    result=model.predict(features).reshape(1,-1)
    st.write("predicted sales::::",result)# if error then result[0].
