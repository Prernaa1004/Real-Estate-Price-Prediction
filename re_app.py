#x=data[["bed","bath","state","house_size"]
import streamlit as st
import numpy as np
import joblib
import pandas as pd

df = pd.read_csv("states_nap.csv")
state_names = df["state"].tolist()


scalar=joblib.load("scalar_re.pkl")
model=joblib.load("best_model.pkl")
st.title("Real Estate price prediction App")
st.divider()

bed=st.number_input("Enter number of bedrooms",value=2,step=1)
bath=st.number_input("Enter number of bathrooms",value=2,step=1)
# names_of_state=
states=st.selectbox("Select the state",state_names)
house_size=st.number_input("Enter the size of the house in square feet",value=1000,step=50)
st.divider()
predict_button=st.button("Predict the price")
st.divider()
if predict_button:
    st.balloons()
    state_price = df[df["state"] == states]["state_price"].values[0]  
    X=[bed,bath,state_price,house_size]
# Extract value

    X1=np.array(X)
    x1_array=scalar.transform([X1])
    price=model.predict(x1_array)
    st.write(f"The price of the house is ${round(price[0])}")
else:
    st.write("Please enter the values to predict the price")


