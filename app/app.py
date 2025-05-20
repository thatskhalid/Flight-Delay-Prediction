import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample



@st.cache_data

def load_data():
    df = pd.read_csv("/Users/khali/Desktop/Coding Workspace/Flight Delay Prediction/data/raw/Airline_Delay_Cause.csv")
    df["high_delay"] = (df["delay_rate"] > 0.2).astype(int)

    x = df[["month" , "carrier" , "airport" , "carrier_ct" , "weather_ct" , "nas_ct" , "late_aircraft_ct" ]]
    y = ["high_delay"]
    x = pd.get_dummies(x , columns = ["carrier" , "airport"] , drop_first = True)


    df_balanced = pd.concat([x,y] axis = 1 )
    df_majority = df_balanced[df_balanced["high_delay"] == 0 ]
    df_minority = df_balanced[df_balanced["high_delay"] == 1 ]

    df_minority_upsampled = resample(df_minority , replace = True , n_samples = len(df_majority) , random_state = 42 )
    df_upsampled = pd.concat([df_majority , df_minority_upsampled])

    x = df_upsampled.drop("high_delay" , axis = 1 )
    y = df_upsampled["high_delay"]
    return x , y


x , y = load_data()
x_train, x_test , y_train, y_test = train_test_split(x,y , test_size = 0.2 , random_state = 42 )

model = RandomForestClassifier(n_estimators=100 , random_state = 42 )
model.fit(x_train,y_train)

st.title("✈️ Flight Delay Risk Predictor")
st.write("This app predicts if a flight will have a **high delay rate** based on airline and cause factors.")


