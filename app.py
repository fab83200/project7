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

st.set_page_config(                                                             # sets a title for the browser tab
    page_title="💰Credit Score API",
    page_icon="💰💵🪙💸💲",)

st.title("💲Project 7💲")
st.header("")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write("""Write the details here:     
-  Detail1 
-  Detail2
	    """)
    st.markdown("")

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
    >>> upload_file("application_train_lite.csv", "application_train")
    """
    variable = pd.read_csv(file, sep=",")
    for tr in variable.describe(include='object').columns:
        variable[tr]=variable[tr].fillna((variable[tr].mode()))
    for ci in variable.describe().columns:
        variable[ci]=variable[ci].fillna((variable[ci].median()))
    return variable
    
# This is the main train table, with TARGET
app_train = upload_file("application_train_lite.csv", "application_train")

# This is the main test table, without TARGET
app_test = upload_file("application_test_lite.csv", "application_test")

# This cell is used to label encode all non numerical features for the `app_train` and `app_test` datasets
l = LabelEncoder()
for p in app_train.describe(include='object').columns:
  app_train[p]=l.fit_transform(app_train[p])
# l = LabelEncoder()
for q in app_test.describe(include='object').columns:
  app_test[q]=l.fit_transform(app_test[q])


# Prepare the Datasets
x = app_train.drop(['TARGET', 'SK_ID_CURR'],axis=1)
y = app_train['TARGET']
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=.30, random_state=42)
x_test = app_test.drop(['SK_ID_CURR'],axis=1)
x_valid = x_valid.reset_index()
del x_valid['index']
feature_names = x_train.columns

# Choose a random sample
idx = random.randint(1, len(x_valid))
#idx = 18530

random_element = x_valid.loc[idx]

st.markdown("### Select your desired loan parameters")

form = st.form(key="my_form")

with form:
	cols = st.columns((1, 1))
	amt_credit = cols[0].number_input("What is the desired loan amount:",       		# Name of the number_input
						   key='amt_credit_widget',                                		# Name of the variable for the data
						   value=float(x_valid.loc[idx]['AMT_CREDIT']),					# Sets the default value
						   help=f"Choose a number between {x['AMT_CREDIT'].min():,} and {x['AMT_CREDIT'].max():,}", 
						   on_change=None)                                      		# Name of the function to use `on_change`,

	amt_annuity = cols[1].number_input("What is the desired yearly repayment?", 	    # Name of the number_input
						   key='amt_annuity_widget',                            		# Name of the variable for the data
						   value=float(x_valid.loc[idx]['AMT_ANNUITY']),        		# Sets the default value
						   help=f"Choose a number between {x['AMT_ANNUITY'].min():,} and {x['AMT_ANNUITY'].max():,}", 
						   on_change=None)                                      		# Name of the function to use `on_change`,

	birthday = st.date_input("What is your birthday?",                       			# Name for the birthday variable
						   key='birthday_widget')                                      	# Name of the variable for the data

	ext_source_1 = cols[0].number_input("What is the ext_source_1",              		# Name of the number_input
						   key='ext_source_1_widget',                                  	# Name of the variable for the data
						   value=float(x_valid.loc[idx]['EXT_SOURCE_1']),       		# Sets the default value
						   help=f"Choose a number between {x['EXT_SOURCE_1'].min():,} and {x['EXT_SOURCE_1'].max():,}", 
						   on_change=None)                                      		# Name of the function to use `on_change`,

	ext_source_3 = cols[1].number_input("What is the ext_source_3",             		# Name of the number_input
						   key='ext_source_3_widget',                                  	# Name of the variable for the data
						   value=float(x_valid.loc[idx]['EXT_SOURCE_3']),       		# Sets the default value
						   help=f"Choose a number between {x['EXT_SOURCE_3'].min():,} and {x['EXT_SOURCE_3'].max():,}", 
						   on_change=None)                                      		# Name of the function to use `on_change`,

	random_element[6] = amt_credit
	random_element[7] = amt_annuity
	st.write(f"The birthday is {birthday}")
	#random_element[15] = birthday
	random_element[39] = ext_source_1
	random_element[41] = ext_source_3
	payment_rate = amt_annuity / amt_credit

	submit_button = st.form_submit_button(label="Submit")

	
# standardizes and normalizes the x data
std_scale = StandardScaler().fit(x_train)                       	              
x_train = std_scale.transform(x_train)
x_valid = std_scale.transform(x_valid)
x_test = std_scale.transform(x_test)
random_element = std_scale.transform(np.array(random_element).reshape(1, -1))

# Perform a Logistic Regression
lr = LogisticRegression(max_iter=3000)
lr.fit(x_train, y_train)                                                        # (215257, 120)   # (215257,)
y_pred_lr = lr.predict(x_valid)                                                 # (92254,)        # (92254, 120)
y_pred_lr_idx  = lr.predict(random_element)[0]
probability  = lr.predict_proba(random_element)[0, 1]
    
# Lime Instanciation
explainer = lime_tabular.LimeTabularExplainer(np.array(x_train), mode="classification",
                                              class_names=np.array(['normal', 'default']),
                                              feature_names=np.array(feature_names))
explanation = explainer.explain_instance(x_valid[idx], lr.predict_proba, num_features=10)

if submit_button:
	st.write(f"The chosen parameters give the following results:")
	st.write("Prediction : ", y_pred_lr_idx)
	st.write(f'Probablility of being a Defaulter: {probability:.2%}')
	# Display explainer HTML object
	components.html(explanation.as_html(), height=800)

#if st.button("Explain Results"):
#    with st.spinner('Calculating...'):
        


       
