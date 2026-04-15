import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_all():
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans.pkl')
    encoder = load_model('encoder.keras')
    data = pd.read_csv('rfm_data.csv')
    return scaler, kmeans, encoder, data

scaler, kmeans, encoder, data = load_all()

st.title("Customer Segmentation System")

# -------------------------------
# Input Section
# -------------------------------
st.header("Enter Customer Data")

recency = st.number_input("Recency (days)", min_value=0)
frequency = st.number_input("Frequency", min_value=0)
monetary = st.number_input("Monetary Value", min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Segment"):

    input_data = np.array([[recency, frequency, monetary]])

    scaled = scaler.transform(input_data)
    encoded = encoder.predict(scaled)
    cluster = kmeans.predict(encoded)[0]

    st.success(f"Cluster: {cluster}")

    # Segment labels
    if cluster == 0:
        st.write("Segment: Low Value")
    elif cluster == 1:
        st.write("Segment: Regular")
    elif cluster == 2:
        st.write("Segment: High Value")
    else:
        st.write("Segment: Premium")

# -------------------------------
# Show Dataset
# -------------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(data.head())

# -------------------------------
# Cluster Insights
# -------------------------------
if st.checkbox("Show Cluster Insights"):
    st.write(data.groupby('Cluster').mean())

# -------------------------------
# Similar Customers
# -------------------------------
if st.checkbox("Find Similar Customers"):

    cust_id = st.number_input("Enter Customer ID")

    if st.button("Find Similar"):

        from sklearn.metrics.pairwise import cosine_similarity

        features = data[['Recency','Frequency','Monetary']]
        sim = cosine_similarity(features)

        idx = data[data['CustomerID'] == cust_id].index

        if len(idx) == 0:
            st.error("Customer not found")
        else:
            idx = idx[0]
            scores = sim[idx]
            similar_idx = scores.argsort()[::-1][1:6]

            st.write(data.iloc[similar_idx])
