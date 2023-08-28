import streamlit as st
import pickle
import numpy as np

# Load the classifier and pipeline from a file
with open('classifier1.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.title('Fraud Detection')


import streamlit as st
import pandas as pd
import numpy as np

# Define the columns and their descriptions
columns = {
    'step': 'maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).',
    'type': 'CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.',
    'amount': 'amount of the transaction in local currency.',
    'nameOrig': 'customer who started the transaction',
    'oldBalanceOrig': 'initial balance before the transaction',
    'newBalanceOrig': 'new balance after the transaction.',
    'nameDest': 'customer who is the recipient of the transaction',
    'oldBalanceDest': 'initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).',
    'newBalanceDest': 'new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).'
}

# Set default values for the input fields
data = ['1','TRANSFER','9839.64','C1231006815','170136.0','160296.3','M1979787155','0.0','0.0']

# Create input fields for each column
inputs = {}
for i, (col, desc) in enumerate(columns.items()):
    st.write(f'{col}: {desc}')
    if col == 'type':
        inputs[col] = st.selectbox(col, ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], index=['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'].index(data[i]))
    else:
        inputs[col] = st.text_input(col, value=data[i])
# Create a button to make a prediction
if st.button('Predict'):
    # Create a new dataframe with the input data
    data = [inputs[col] for col in columns.keys()]
    df = pd.DataFrame([data], columns=columns.keys())

    # Convert the columns to the desired data types
    df['step'] = df['step'].astype('int64')
    df['type'] = df['type'].astype('object')
    df['amount'] = df['amount'].astype('float64')
    df['nameOrig'] = df['nameOrig'].astype('object')
    df['oldBalanceOrig'] = df['oldBalanceOrig'].astype('float64')
    df['newBalanceOrig'] = df['newBalanceOrig'].astype('float64')
    df['nameDest'] = df['nameDest'].astype('object')
    df['oldBalanceDest'] = df['oldBalanceDest'].astype('float64')
    df['newBalanceDest'] = df['newBalanceDest'].astype('float64')

    if data[1]=='TRANSFER' or data[1] == 'CASH_OUT':
        X = df
        st.write(X)
        # Eliminate columns shown to be irrelevant for analysis in the EDA
        X = X.drop(['nameOrig', 'nameDest'], axis=1)

        # Binary-encoding of labelled data in 'type'
        X.loc[X.type == 'TRANSFER', 'type'] = 0
        X.loc[X.type == 'CASH_OUT', 'type'] = 1
        X.type = X.type.astype(int) # convert dtype('O') to dtype(int)
        X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), ['oldBalanceDest', 'newBalanceDest']] = - 1
        X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), ['oldBalanceOrig', 'newBalanceOrig']] = np.nan
        X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
        X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
        a=classifier.predict_proba(X.loc[[0]])
        a1,a2=float(a[:,0]),float(a[:,1])
        if(a1>a2):
            st.write("No Fraud in the Transaction")
        else:
            st.write("Fraud in the Transaction")
    else:
        st.write('No Fraud in the Tranaction Founded')
