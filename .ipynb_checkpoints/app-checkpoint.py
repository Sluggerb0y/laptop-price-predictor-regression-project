
import streamlit as st
import pickle
import pandas as pd

pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

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
