import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

@st.cache_resource
def train_model():
    df = pd.read_csv("laptop_data.csv")

    X = df.drop(columns=["Price"])
    y = df["Price"]

    cat_cols = ["Company", "TypeName", "Cpu brand", "Gpu brand", "os"]

    step1 = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    step2 = XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        random_state=3,
        n_jobs=1
    )

    pipe = Pipeline([
        ("step1", step1),
        ("step2", step2)
    ])

    pipe.fit(X, y)
    return pipe, df

pipe, df = train_model()

st.title("Laptop Price Predictor")

company = st.selectbox("Company", sorted(df["Company"].unique()))
typename = st.selectbox("Laptop Type", sorted(df["TypeName"].unique()))
ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
weight = st.number_input("Weight (kg)", value=1.5)

touchscreen = st.selectbox("Touchscreen", [0, 1])
ips = st.selectbox("IPS Display", [0, 1])
ppi = st.number_input("PPI", value=141.21)

cpu = st.selectbox("CPU Brand", sorted(df["Cpu brand"].unique()))
gpu = st.selectbox("GPU Brand", sorted(df["Gpu brand"].unique()))
os = st.selectbox("OS", sorted(df["os"].unique()))

hdd = st.selectbox("HDD (GB)", sorted(df["HDD"].unique()))
ssd = st.selectbox("SSD (GB)", sorted(df["SSD"].unique()))

if st.button("Predict Price"):
    query = pd.DataFrame([{
        "Company": company,
        "TypeName": typename,
        "Ram": ram,
        "Weight": weight,
        "Touchscreen": touchscreen,
        "Ips": ips,
        "ppi": ppi,
        "Cpu brand": cpu,
        "first": 0,
        "second": 0,
        "Layer1HDD": 0,
        "Layer1SSD": ssd,
        "Layer1Hybrid": 0,
        "Layer1Flash_Storage": 0,
        "Layer2HDD": 0,
        "Layer2SSD": 0,
        "Layer2Hybrid": 0,
        "Layer2Flash_Storage": 0,
        "HDD": hdd,
        "SSD": ssd,
        "Gpu brand": gpu,
        "os": os
    }])

    prediction = pipe.predict(query)[0]
    st.success(f"Predicted Laptop Price: ₹{round(prediction, 2)}")