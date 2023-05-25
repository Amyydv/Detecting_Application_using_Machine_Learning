import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import os

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import pycaret
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment

st.title("Detecting Application using Machine Learning")

if os.path.exists("sourcev.csv"):
    df = pd.read_csv("sourcev.csv",index_col=None)

with st.sidebar:
    st.header("Welcome to the Application!")
    st.info("Select the options to work on the dataset. The uploded dataset can be analysed by using 'Analysis' button.\n To train the model, choose the area you want to work on i.e., Classification & Regression. \n You can download the model (.pkl) file using 'Download' button. ")
    st.caption("Do try it out!")
    choose=st.radio(":computer:",["Dataset","Analysis","Training","Download"])
    
if choose=="Dataset":
    st.caption("This model is based on the backdrop of classification and Regression alogithum of machine learning. It helps in analysing and exploring your dataset in a smart way. cheers. 😜 ")
    st.write("Please upload your dataset here. Only .csv files allowed!")
    dataset_value = st.file_uploader("Upload here")
    
    if dataset_value:
        df = pd.read_csv(dataset_value, index_col=None)
        df.to_csv("sourcev.csv", index = None)
        st.dataframe(df)

if choose=="Analysis":
    st.subheader("Perform profiling on Dataset. It will statistically analyse your data. I have used Pandas profiling for analysis.")
    if st.sidebar.button("Do Analysis"):
        profile_report = df.profile_report() 
        st_profile_report(profile_report)
    
if choose=="Training":
    st.header("Start Training your Model now. Please select your technique based on the choices given below.")
    choice = st.sidebar.selectbox("Select your Technique:", ["Classification","Regression"])
    target = st.selectbox("Select you Target Variable",df.columns)
    if choice=="Classification":
        if st.sidebar.button("Classification Train"):
            s1 = ClassificationExperiment()
            s1.setup(data=df, target=target)
            setup_df = s1.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
            
            best_model1 = s1.compare_models()
            compare_model = s1.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
            
            best_model1
            s1.save_model(best_model1,"Machine Learning Model")
    else:
        if st.sidebar.button("Regression Train"):
            s2 = RegressionExperiment()
            s2.setup(data=df, target=target)
            setup_df = s2.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
            
            best_model2 = s2.compare_models()
            compare_model = s2.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
            
            best_model2
            s2.save_model(best_model2,"Machine Learning Model")

if choose =="Download":
    with open("Machine Learning model.pkl",'rb') as f:
        st.caption("The model will be downloaded using .pkl file. Download your model from here:")
        st.download_button("Download the file",f,"Machine Learning model.pkl")
