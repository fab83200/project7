#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import random

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import accuracy_score
#from lime import lime_tabular
#import streamlit.components.v1 as components

title_text = 'LIME Explainer Dashboard for credit score'

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
st.text("")

../dataset/estadistical.csv
pd.read_csv("../dataset/estadistical.csv")

# This is the main train table, with TARGET
app_train = pd.read_csv("../application_train_lite.csv", sep=",")
for tr in app_train.describe(include='object').columns:
    app_train[tr]=app_train[tr].fillna((app_train[tr].mode()))
for ci in app_train.describe().columns:
    app_train[ci]=app_train[ci].fillna((app_train[ci].median()))
