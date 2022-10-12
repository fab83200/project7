import streamlit as st
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from lime import lime_tabular
import streamlit.components.v1 as components

title_text = 'Credit Score API'

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
st.text("")

def upload_file(file, variable):
    """upload_file function to load a .csv data file into a DataFrame.
        If 'object' values for a feature are missing,
        it is replaced by the mode of that feature (ie. the most common feature).
        If 'numeric' values for a feature are missing,
        it is replaced by the median of that feature.

    Parameters
    ----------
    file: str
        The name of the csv file to upload. It must be under the root of the repository.
    variable: str
        The name of the variable under which the Dataframe will be stored
    
    Returns
    -------
    A pandas type DataFrame with no missing values.
    
    Example
    -------
    >>> upload_file(application_train_lite.csv, application_train)
    """
    variable = pd.read_csv(file, sep=",")
    for tr in variable.describe(include='object').columns:
        variable[tr]=variable[tr].fillna((variable[tr].mode()))
    for ci in variable.describe().columns:
        variable[ci]=variable[ci].fillna((variable[ci].median()))
    return variable
    
# This is the main train table, with TARGET
upload_file("application_train_lite.csv", "application_train")
#variable = pd.read_csv("application_train_lite.csv", sep=",")

# This is the main test table, without TARGET
#upload_file(application_test_lite.csv, application_test)

application_train


    
    
    
    
    
    
    
