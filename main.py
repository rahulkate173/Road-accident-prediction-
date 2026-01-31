import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Build correct absolute paths
base_dir = os.path.dirname(__file__)  # current directory
model_path = os.path.join(base_dir, "models", "xgboost_model.pkl")
scalar_path = os.path.join(base_dir, "models", "standard_scalar.pkl")

# Load the files
model = joblib.load(model_path)
scalar = joblib.load(scalar_path)

def feature_engineering(train_df):
    # ratio and product
    train_df['Density_Index'] = train_df['speed_limit'] / train_df['num_lanes']
    train_df['Complex_Risk'] = train_df['curvature'] * train_df['speed_limit']
    train_df['Accident_Rate'] = train_df['num_reported_accidents'] / train_df['num_lanes']
    # Binning speed_limit
    bins = [0, 30, 60, 100] # Define bins for speed limits
    labels = ['Low_Speed', 'Medium_Speed', 'High_Speed']
    train_df['Speed_Category'] = pd.cut(
        train_df['speed_limit'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    # Log transform 
    train_df['log_curvature'] = np.log1p(train_df['curvature'])
    # Square root transform
    train_df['sqrt_num_lanes'] = np.sqrt(train_df['num_lanes'])
    return train_df

def encode(train_df):
    categorical_cols = train_df.select_dtypes(include=['object','category']).columns
    temp_df = train_df.copy(deep=True)
    encoded_df = pd.get_dummies(train_df,columns=categorical_cols,drop_first=True)
    return encoded_df

def model_process(df, model, scalar):
    df = feature_engineering(df)
    df = encode(df)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df.loc[:, numerical_cols] = scalar.transform(df[numerical_cols])
    model_features = model.get_booster().feature_names  

    # Add missing columns (set to 0)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Drop any extra columns that model doesn't expect
    df = df[model_features]
    y_pred = model.predict(df)
    return f"{y_pred[0]:.2f}"


## web app
st.markdown("# Road Prediction App")
st.markdown("> An app to predict the accident risk train ")
## the input for the model 
road_type = st.selectbox("Type of Road on which driven",['urban','rural','highway'])
num_lanes = st.selectbox("Number of lanes road has ",[1,2,3,4])
curvature = st.number_input('Curvature of road',min_value=0.000000,max_value=1.000000)
speed_limit = st.number_input("Speed limit on the road",min_value=25.000000,max_value=70.000000)
lighting = st.selectbox("Lighting condition while driving",['daylight','dim','night'])
weather = st.selectbox("weather condition while driving",['rainy','clear','foggy'])
road_signs_present = st.selectbox("Are road signs present while driving ?",[True,False])
public_road = st.selectbox("Are u driving on public road ? ",[True,False])
time_of_day = st.selectbox("Time of day u are driving? ",['afternoon','evening','morning'])
holiday = st.selectbox("Is holiday on that day?",[True,False])
school_season = st.selectbox("So does school season is active?",[True,False])
num_reported_accidents = st.slider("Number of reported accident on that street?",0,6,0)
##---------input completed---------------
## Building the input
df = pd.DataFrame({
    "road_type": [road_type],
    "num_lanes": [num_lanes],
    "curvature": [curvature],
    "speed_limit": [speed_limit],
    "lighting": [lighting],
    "weather": [weather],
    "road_signs_present": [road_signs_present],
    "public_road": [public_road],
    "time_of_day": [time_of_day],
    "holiday": [holiday],
    "school_season": [school_season],
    "num_reported_accidents": [num_reported_accidents]
})

##model processing and output prediction 
if st.button("Predict Accident Risk"):
    try:
        prediction = model_process(df, model, scalar)
        st.markdown("### 🚦 Accident Risk Prediction:")
        st.success(f"Predicted Risk Level: **{float(prediction)*100}%**")
        st.caption("Model trained on over **5 lakh data samples**.")
        st.caption("Dataset Link : https://www.kaggle.com/competitions/playground-series-s5e10 ")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
