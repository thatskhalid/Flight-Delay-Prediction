import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

model = joblib.load("/Users/khali/Desktop/Coding Workspace/Flight Delay Prediction/model/model.pkl")
feature_columns = joblib.load("/Users/khali/Desktop/Coding Workspace/Flight Delay Prediction/model/feature_columns.pkl")

@st.cache_data

def load_data():
    df = pd.read_csv("/Users/khali/Desktop/Coding Workspace/Flight Delay Prediction/data/processed/Airline_Delay_Cause_Clean.csv")
    df["high_delay"] = (df["delay_rate"] > 0.2).astype(int)

    x = df[["month" , "carrier" , "airport" , "carrier_ct" , "weather_ct" , "nas_ct" , "late_aircraft_ct" ]]
    y = df["high_delay"]
    x = pd.get_dummies(x , columns = ["carrier" , "airport"] , drop_first = True)



    df_balanced = pd.concat([x, y] , axis = 1)
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

st.subheader("Feature Importances")
importances = model.feature_importances_
feature_names = x.columns
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(range(len(importances)) , importances[indices])
ax.set_xticks(range(len(importances)))
ax.set_xticklabels(feature_names[indices] , rotation = 90)
st.pyplot(fig)


st.subheader("Predict Delay Risk")


carriers = [col.replace("carrier_" , "") for col in x.columns if "carrier_" in col]
airports = [col.replace("airports_" , "") for col in x.columns if "airports_" in col]

month = st.slider("Month" , 1 , 12 , 1)
carrier = st.selectbox("Carrier" , carriers)
airport = st.selectbox("Airport" , airports)
carrier_ct = st.number_input("Carrier-caused delays" , 0.0 , 100.0, 10.0 )
weather_ct = st.number_input("Weather-caused delays" , 0.0 , 100.0 , 5.0 )
nas_ct = st.number_input("NAS-caused delays" , 0.0 , 100.0 , 8.0)
late_aircraft_ct = st.number_input("Late aircraft delays" , 0.0 , 100.0 , 7.0)


input_data = {
    "month" : [month],
    "carrier_ct" : [carrier_ct],
    "weather_ct" : [weather_ct],
    "nas_ct" : [nas_ct],
    "late_aircraft_ct" : [late_aircraft_ct] 
}


for c in carriers:
    input_data[f"carrier_{c}"] = [1 if c == carrier else 0]

for a in airports:
    input_data[f"airport_{a}"] = [1 if a == airport else 0 ]

input_df = pd.DataFrame(input_data)

input_df = input_df.reindex(columns = feature_columns , fill_value=0)


if st.button("Predict Delay Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High Delay Risk")
    else:
        st.success("✅ Low Delay Risk")



