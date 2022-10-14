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
    page_title="💰Credit Score API")

st.title("💲Project 7: Local Analysis💲")
st.header("")

with st.expander("# ℹ️ - About this app", expanded=True):
    st.write("""This API will make a prediction on loan default with parameters choosen by the user.    
For each case, there are too many parameters needed. Therefore, we made specific assumptions and choices:
-  Most users are only interested in 3 fields (`Value of property`, `downpayment`, `credit length`),
-  Our Python notebook analyzis showed that the `household incomes` matter,
-  For the purpose of this OC Project, we decided to add 2 customs radio button for the sake of diversity,
-  All the other missing variables will be filled up with their dataset medians (only numeric data with LabelEncoder). 

Below, select your desired loan parameters and hit the button `SUBMIT`.

Then 2 types of computations is made:
-  The first one is a classic Logistic Regression algorithm to determine the probability of the user to be a defaulter,
-  The second one is performing a feature importance using the LIME library. 

NOTA: _My Github account sets a file size limit at 25 MB, so I'm only using 1/6th of the total dataset._
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

last_index = x_valid.shape[0]
x_valid.loc[last_index] = x_valid.loc[last_index-1]
for tr in x_valid.describe().columns:
  x_valid.loc[last_index][tr] = x_valid[tr].median()
#last_element = x_valid.loc[last_index]

st.markdown("### Select your desired loan parameters")

form = st.form(key="my_form")

with form:
	cols=st.columns(2)
	amt_goods_price = cols[0].number_input("What is the value of the property (USD):",  	# Name of the number_input
						   key='amt_goods_price_widget')
	
	down_payment = cols[1].number_input("What is amount of down payment (USD):",  			# Name of the number_input
						   key='down_payment_widget') 
	
	amt_annuity = cols[0].number_input("What is the desired credit length?", 	    		# Name of the number_input
						   key='amt_annuity_widget',                            			# Name of the variable for the data
						   value=20.0, 														# Sets the default value
						   help="Choose a number between 1 and 30 years")
	
	amt_income_total = cols[1].number_input("What is Your total income (USD)?", 			# Name of the number_input
						   key='amt_income_total')
	
	genders_available = ['MALE', 'FEMALE']
	features_available = ['YES', 'NO']
	
	code_gender = cols[0].radio("What is your gender?",                          			# creates a Radio_Buttons widget
							genders_available,                               		  		# set the available options
							key="flag_own_car_button")                         				# names the radio button

	flag_own_car = cols[1].radio("Do you own a car?",                          				# creates a Radio_Buttons widget
							features_available,                               		  		# set the available options
							key="flag_own_car_button")                         				# names the radio button

	x_valid.loc[last_index]['CODE_GENDER'] = code_gender
	x_valid.loc[last_index]['FLAG_OWN_CAR'] = flag_own_car
	x_valid.loc[last_index]['AMT_INCOME_TOTAL'] = amt_income_total
	x_valid.loc[last_index]['AMT_CREDIT'] = amt_goods_price - down_payment
	x_valid.loc[last_index]['AMT_ANNUITY'] = amt_annuity
	x_valid.loc[last_index]['AMT_GOODS_PRICE'] = amt_goods_price

	submit_button = st.form_submit_button(label="Submit")
	

if submit_button:
	with st.spinner('Calculating, it takes about 1 minute...'):
		# standardizes and normalizes the x data
		std_scale = StandardScaler().fit(x_train)                       	              
		x_train = std_scale.transform(x_train)
		x_valid = std_scale.transform(x_valid)
		x_test = std_scale.transform(x_test)
		last_element = x_valid[last_index].reshape(1, -1)
		#random_element = std_scale.transform(np.array(random_element).reshape(1, -1))

		# Perform a Logistic Regression
		lr = LogisticRegression(max_iter=3000)
		lr.fit(x_train, y_train)                                                        # (215257, 120)   # (215257,)
		y_pred_lr = lr.predict(x_valid)                                                 # (92254,)        # (92254, 120)
		y_pred_lr_idx  = lr.predict(last_element)[0]
		probability  = lr.predict_proba(last_element)[0, 1]

		# Lime Instanciation
		explainer = lime_tabular.LimeTabularExplainer(np.array(x_train), mode="classification",
													  class_names=np.array(['normal', 'default']),
													  feature_names=np.array(feature_names))
		explanation = explainer.explain_instance(x_valid[last_index], lr.predict_proba, num_features=10)
		
		st.markdown("### The chosen parameters give the following results")
		st.write("Prediction : ", y_pred_lr_idx)
		st.write(f'Probablility of being a Defaulter: {probability:.2%}')
		st.markdown("")
		# Display explainer HTML object
		st.markdown("### Explanation with Lime")
		components.html(explanation.as_html(), height=800)
        
